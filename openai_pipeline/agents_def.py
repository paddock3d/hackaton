"""
Agent definitions for the BOM Smart Processor pipeline.
Agents orchestrated via OpenAI Agents SDK with guardrails,
vision support, and native MCP server connections.
"""
import os
import json
from agents import (
    Agent, FileSearchTool,
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput,
)
from tools import catalog_mcp

INSTRUCTIONS_DIR = os.path.join(os.path.dirname(__file__), "agent_instructions")
VECTOR_STORE_FILE = os.path.join(os.path.dirname(__file__), ".vector_store_id")


def _load_instructions(filename: str) -> str:
    with open(os.path.join(INSTRUCTIONS_DIR, filename)) as f:
        return f.read()


def _get_vector_store_id() -> str:
    vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    if vs_id:
        return vs_id
    if os.path.exists(VECTOR_STORE_FILE):
        with open(VECTOR_STORE_FILE) as f:
            return f.read().strip()
    raise RuntimeError(
        "No vector store ID found. Run setup_vector_store.py first "
        "or set OPENAI_VECTOR_STORE_ID env var."
    )


# ── Guardrails ──

async def _validate_bom_input(ctx, agent, input):
    """Input guardrail: validate BOM data is present and well-formed."""
    text = input if isinstance(input, str) else json.dumps(input)
    issues = []
    if len(text.strip()) < 20:
        issues.append("Input too short — no BOM data detected")
    bom_keywords = ["part", "qty", "quantity", "description", "bom", "component",
                     "resistor", "capacitor", "mcu", "sensor", "ic", "connector"]
    has_keyword = any(k in text.lower() for k in bom_keywords)
    if not has_keyword:
        issues.append("No BOM-related content found (missing part numbers, component names, or quantities)")
    if issues:
        return GuardrailFunctionOutput(
            output_info={"blocked": True, "reasons": issues},
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info={"blocked": False, "checks_passed": ["format", "content"]}, tripwire_triggered=False)

bom_input_guardrail = InputGuardrail(guardrail_function=_validate_bom_input, name="bom_format_validator")


async def _validate_quote_output(ctx, agent, output):
    """Output guardrail: validate the quote JSON is complete and reasonable."""
    text = output if isinstance(output, str) else str(output)
    issues = []
    try:
        data = json.loads(text)
        items = data.get("line_items", data.get("matched_items", []))
        fs = data.get("financial_summary", {})
        if not items:
            issues.append("Quote has no line items")
        if not fs:
            issues.append("Quote missing financial summary")
        total = fs.get("grand_total_required", 0)
        if isinstance(total, (int, float)) and total > 100000:
            issues.append(f"Grand total ${total:,.2f} exceeds $100K — verify quantities")
        zero_price = [i for i in items if i.get("line_total", 0) == 0 or i.get("unit_price", 0) == 0]
        if zero_price:
            issues.append(f"{len(zero_price)} items have $0 pricing")
    except (json.JSONDecodeError, TypeError):
        issues.append("Output is not valid JSON")

    if issues:
        return GuardrailFunctionOutput(
            output_info={"blocked": True, "reasons": issues},
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(
        output_info={"blocked": False, "checks_passed": ["json_valid", "has_items", "has_financials", "pricing_reasonable"]},
        tripwire_triggered=False,
    )

quote_output_guardrail = OutputGuardrail(guardrail_function=_validate_quote_output, name="quote_completeness_validator")


# ── Vision Agent (BOM image → CSV) ──

bom_vision_agent = Agent(
    name="BOM_Vision_Extractor",
    model="gpt-4.1",
    instructions="""You are a BOM image parser. You receive an image of a Bill of Materials (handwritten, printed, screenshot, or PDF scan).

TASK: Extract ALL parts from the image into a clean CSV format.

OUTPUT FORMAT (exactly this, no markdown fences):
part_number,description,quantity
VALUE1,VALUE2,VALUE3
...

RULES:
- Extract every row from the BOM table/list
- Infer part numbers from component markings, text, or labels
- If quantity is not visible, default to 1
- If part number is unclear, use the component description as the part number
- Output ONLY the CSV — no explanations, no markdown, no headers other than the CSV header
- If you cannot read the image or it's not a BOM, output: ERROR: Could not extract BOM data from image""",
)


# ── Pipeline Agents ──

bom_parser_and_matcher = Agent(
    name="BOM_Parser_Product_Matcher",
    model="gpt-4.1",
    instructions=_load_instructions("parser_and_product_matcher_agent.txt"),
    mcp_servers=[catalog_mcp],
    input_guardrails=[bom_input_guardrail],
)

datasheet_specialist = Agent(
    name="Datasheet_Specialist",
    model="gpt-4.1",
    instructions=_load_instructions("datasheet_specialist_openai.txt"),
    tools=[
        FileSearchTool(
            vector_store_ids=[_get_vector_store_id()],
            max_num_results=5,
            include_search_results=True,
        ),
    ],
)

quote_generator = Agent(
    name="Quote_Generator",
    model="gpt-4.1",
    instructions=_load_instructions("quote_generator_agent.txt"),
    mcp_servers=[catalog_mcp],
    output_guardrails=[quote_output_guardrail],
)

orchestrator = Agent(
    name="BOM_Procurement_Orchestrator",
    model="gpt-4.1",
    instructions="""You orchestrate BOM processing through 3 specialist agents in strict sequence.
Your ONLY job is to pass data between agents. Do NOT modify, summarize, or interpret the data.

SEQUENCE:
1. Hand off to "BOM_Parser_Product_Matcher" with the user's input (the S3 file path).
   Wait for it to return JSON with parsed BOM and matched products.

2. Take the COMPLETE JSON output from step 1 and hand it off to "Datasheet_Specialist".
   Wait for it to return enriched JSON with technical specs added.

3. Take the COMPLETE JSON output from step 2 and hand it off to "Quote_Generator".
   Wait for it to return the final quote JSON.

4. Return the Quote Generator's output EXACTLY as-is. Do not add any text or modify it.

RULES:
- Pass the FULL JSON between agents. Do not truncate or summarize.
- Do not call any tools yourself.
- Do not add commentary, explanations, or formatting.
- If any agent returns an error JSON, return that error JSON immediately.""",
    handoffs=[bom_parser_and_matcher, datasheet_specialist, quote_generator],
)
