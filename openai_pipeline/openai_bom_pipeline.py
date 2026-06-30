#!/usr/bin/env python3
"""
OpenAI Agents SDK pipeline for BOM Smart Processor.

Features:
    - 3-agent pipeline (Parser+Matcher → Datasheet → Quote)
    - Vision support (image/PDF BOM upload → CSV extraction)
    - Input guardrails (BOM format validation)
    - Output guardrails (quote completeness validation)
    - Tracing (OpenAI dashboard link)

Usage:
    python3 openai_bom_pipeline.py <file_path>

    Called from SAP CAP backend via child_process.spawn.
    Outputs the final quote JSON to stdout.
"""
import asyncio
import base64
import json
import mimetypes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import Runner, trace, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from agents_def import bom_parser_and_matcher, datasheet_specialist, quote_generator, bom_vision_agent
from tools import catalog_mcp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
PDF_EXTENSIONS = {".pdf"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in PDF_EXTENSIONS


def _build_vision_input(file_path: str) -> list:
    """Build multi-modal input message for the vision agent."""
    with open(file_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()

    ext = os.path.splitext(file_path)[1].lower()
    mime = mimetypes.guess_type(file_path)[0] or f"image/{ext.lstrip('.')}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract the Bill of Materials from this image into CSV format. Include all parts with part_number, description, and quantity columns."},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                    "detail": "high",
                },
            ],
        }
    ]


def _extract_usage(run_result) -> dict:
    """Extract token usage from a Runner.run result."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        for item in run_result.raw_responses:
            u = getattr(item, "usage", None)
            if u:
                usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
    except Exception:
        pass
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


async def process_bom(file_path: str) -> dict:
    """Run the full BOM processing pipeline with vision, guardrails, and tracing."""
    import time as _time

    pipeline_meta = {
        "vision_used": False,
        "guardrails": {
            "input": {"status": "passed", "checks": []},
            "output": {"status": "passed", "checks": []},
        },
        "trace_id": None,
        "trace_url": None,
        "trace_spans": [],
    }

    pipeline_start = _time.time()

    with trace("BOM Pipeline") as t:
        pipeline_meta["trace_id"] = t.trace_id
        pipeline_meta["trace_url"] = f"https://platform.openai.com/traces/{t.trace_id}"
        print(f"Trace: {pipeline_meta['trace_url']}", file=sys.stderr)

        # ── Vision pre-processing (image/PDF → CSV) ──
        if os.path.isfile(file_path) and (_is_image(file_path) or _is_pdf(file_path)):
            pipeline_meta["vision_used"] = True
            print("Stage 0/3: Vision — extracting BOM from image...", file=sys.stderr)

            t0 = _time.time()
            vision_input = _build_vision_input(file_path)
            vision_result = await Runner.run(bom_vision_agent, input=vision_input)
            bom_content = vision_result.final_output
            pipeline_meta["trace_spans"].append({
                "agent": "BOM_Vision_Extractor",
                "model": "gpt-4.1",
                "duration_s": round(_time.time() - t0, 1),
                "usage": _extract_usage(vision_result),
                "status": "error" if bom_content.startswith("ERROR:") else "ok",
            })

            if bom_content.startswith("ERROR:"):
                return {
                    "error": bom_content,
                    "pipeline_meta": pipeline_meta,
                }

            print(f"  Vision extracted:\n{bom_content[:200]}...", file=sys.stderr)
            input_msg = (
                f"Parse and process this BOM data. The file was an image, "
                f"extracted via GPT-4.1 Vision.\n\nBOM FILE CONTENT:\n{bom_content}"
            )

        elif os.path.isfile(file_path):
            with open(file_path) as f:
                bom_content = f.read()
            input_msg = (
                f"Parse and process this BOM data. The file name is {os.path.basename(file_path)}.\n\n"
                f"BOM FILE CONTENT:\n{bom_content}"
            )
        else:
            input_msg = f"Load and process this BOM file from S3: {file_path}"

        async with catalog_mcp:
            # ── Stage 1: BOM Parser & Product Matcher (with input guardrail) ──
            print("Stage 1/3: BOM Parser & Product Matcher...", file=sys.stderr)
            t1 = _time.time()
            try:
                result1 = await Runner.run(bom_parser_and_matcher, input=input_msg, max_turns=100)
                pipeline_meta["guardrails"]["input"]["checks"] = ["format", "content"]
                pipeline_meta["trace_spans"].append({
                    "agent": "BOM_Parser_Product_Matcher",
                    "model": "gpt-4.1",
                    "duration_s": round(_time.time() - t1, 1),
                    "usage": _extract_usage(result1),
                    "tools": ["semanticSearch", "getProductDetails", "findAlternatives"],
                    "status": "ok",
                })
            except InputGuardrailTripwireTriggered as e:
                info = e.guardrail_result.output.output_info
                pipeline_meta["guardrails"]["input"] = {
                    "status": "blocked",
                    "reasons": info.get("reasons", ["Input validation failed"]),
                }
                pipeline_meta["trace_spans"].append({
                    "agent": "BOM_Parser_Product_Matcher",
                    "model": "gpt-4.1",
                    "duration_s": round(_time.time() - t1, 1),
                    "status": "guardrail_blocked",
                })
                return {
                    "error": "Input guardrail blocked: " + "; ".join(info.get("reasons", [])),
                    "pipeline_meta": pipeline_meta,
                }

            # ── Stage 2: Datasheet Specialist ──
            print("Stage 2/3: Datasheet Specialist...", file=sys.stderr)
            t2 = _time.time()
            result2 = await Runner.run(
                datasheet_specialist,
                input=f"Enrich these matched products with technical specs from datasheets:\n\n{result1.final_output}",
            )
            pipeline_meta["trace_spans"].append({
                "agent": "Datasheet_Specialist",
                "model": "gpt-4.1",
                "duration_s": round(_time.time() - t2, 1),
                "usage": _extract_usage(result2),
                "tools": ["file_search (65 datasheets)"],
                "status": "ok",
            })

            # ── Stage 3: Quote Generator (with output guardrail) ──
            print("Stage 3/3: Quote Generator...", file=sys.stderr)
            t3 = _time.time()
            try:
                result3 = await Runner.run(
                    quote_generator,
                    input=f"Generate a quote for these enriched products:\n\n{result2.final_output}",
                )
                pipeline_meta["guardrails"]["output"]["checks"] = [
                    "json_valid", "has_items", "has_financials", "pricing_reasonable"
                ]
                pipeline_meta["trace_spans"].append({
                    "agent": "Quote_Generator",
                    "model": "gpt-4.1",
                    "duration_s": round(_time.time() - t3, 1),
                    "usage": _extract_usage(result3),
                    "tools": ["generateQuote"],
                    "status": "ok",
                })
            except OutputGuardrailTripwireTriggered as e:
                info = e.guardrail_result.output.output_info
                pipeline_meta["guardrails"]["output"] = {
                    "status": "warning",
                    "reasons": info.get("reasons", ["Output validation failed"]),
                }
                pipeline_meta["trace_spans"].append({
                    "agent": "Quote_Generator",
                    "model": "gpt-4.1",
                    "duration_s": round(_time.time() - t3, 1),
                    "status": "guardrail_warning",
                })
                return {
                    "warning": "Output guardrail flagged: " + "; ".join(info.get("reasons", [])),
                    "raw_output": result3.final_output if hasattr(result3, 'final_output') else str(e),
                    "pipeline_meta": pipeline_meta,
                }

    pipeline_meta["total_duration_s"] = round(_time.time() - pipeline_start, 1)
    total_input = sum(s.get("usage", {}).get("input_tokens", 0) for s in pipeline_meta["trace_spans"])
    total_output = sum(s.get("usage", {}).get("output_tokens", 0) for s in pipeline_meta["trace_spans"])
    pipeline_meta["total_tokens"] = total_input + total_output
    pipeline_meta["input_tokens"] = total_input
    pipeline_meta["output_tokens"] = total_output
    cost = (total_input / 1_000_000) * 2.00 + (total_output / 1_000_000) * 8.00
    pipeline_meta["estimated_cost_usd"] = round(cost, 4)
    print(f"Pipeline complete. {pipeline_meta['total_duration_s']}s, {pipeline_meta['total_tokens']} tokens, ${pipeline_meta['estimated_cost_usd']:.4f}", file=sys.stderr)

    try:
        result = json.loads(result3.final_output)
    except (json.JSONDecodeError, TypeError):
        result = {"status": "COMPLETED", "raw_output": result3.final_output}

    result["pipeline_meta"] = pipeline_meta
    return result


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python3 openai_bom_pipeline.py <file_path>"}))
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print(json.dumps({"error": "OPENAI_API_KEY not set"}))
        sys.exit(1)

    result = await process_bom(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
