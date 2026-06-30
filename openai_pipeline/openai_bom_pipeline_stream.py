#!/usr/bin/env python3
"""
Streaming version of the BOM pipeline.

Outputs newline-delimited JSON events to stderr for real-time UI updates,
and the final result JSON to stdout.

Event format (one JSON per line on stderr):
  {"event": "agent_start",   "agent": 1, "name": "...", "message": "..."}
  {"event": "tool_call",     "agent": 1, "tool": "semanticSearch", "args": "STM32..."}
  {"event": "tool_result",   "agent": 1, "tool": "semanticSearch", "message": "Found 3 matches"}
  {"event": "agent_complete","agent": 1, "name": "...", "duration_s": 12.3, "tokens": 5000}
  {"event": "guardrail",     "type": "input", "status": "passed"}
  {"event": "done",          "message": "Pipeline complete"}
  {"event": "error",         "message": "..."}
"""
import asyncio
import base64
import json
import mimetypes
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    Runner, trace,
    InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
)
from agents.stream_events import RunItemStreamEvent, AgentUpdatedStreamEvent
from agents_def import bom_parser_and_matcher, datasheet_specialist, quote_generator, bom_vision_agent
from tools import catalog_mcp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
PDF_EXTENSIONS = {".pdf"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in PDF_EXTENSIONS


def _build_vision_input(file_path: str) -> list:
    with open(file_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()
    ext = os.path.splitext(file_path)[1].lower()
    mime = mimetypes.guess_type(file_path)[0] or f"image/{ext.lstrip('.')}"
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract the Bill of Materials from this image into CSV format. Include all parts with part_number, description, and quantity columns."},
            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "high"},
        ],
    }]


def emit(event: dict):
    """Write a JSON event line to stderr (streamed to browser via SSE)."""
    sys.stderr.write(json.dumps(event) + "\n")
    sys.stderr.flush()


def _extract_usage(run_result) -> dict:
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


async def run_agent_streamed(agent, input_msg, agent_num, agent_label):
    """Run a single agent with streaming, emitting events for each tool call."""
    t0 = time.time()
    emit({"event": "agent_start", "agent": agent_num, "name": agent_label,
          "message": f"Starting {agent_label}..."})

    result = Runner.run_streamed(agent, input=input_msg, max_turns=100)
    tool_count = 0
    last_tool = ""

    try:
        async for event in result.stream_events():
            if isinstance(event, RunItemStreamEvent):
                if event.name == "tool_called":
                    tool_name = getattr(event.item, "tool_name", None) or "file_search"
                    tool_count += 1
                    last_tool = tool_name
                    emit({"event": "tool_call", "agent": agent_num,
                          "tool": tool_name,
                          "message": f"Calling {tool_name}..."})
                elif event.name == "tool_output":
                    raw = getattr(event.item, "output", "")
                    output_str = str(raw)[:120] if raw else ""
                    emit({"event": "tool_result", "agent": agent_num,
                          "tool": last_tool,
                          "message": output_str})
                elif event.name == "reasoning_item_created":
                    text = ""
                    try:
                        for s in getattr(event.item, "raw_item", {}).get("summary", []):
                            text = s.get("text", "")
                            break
                    except Exception:
                        pass
                    if text:
                        emit({"event": "thinking", "agent": agent_num,
                              "message": text[:150]})
    except MaxTurnsExceeded:
        emit({"event": "thinking", "agent": agent_num,
              "message": f"Max turns reached after {tool_count} tool calls, finalizing..."})

    duration = round(time.time() - t0, 1)
    usage = _extract_usage(result)
    emit({"event": "agent_complete", "agent": agent_num, "name": agent_label,
          "duration_s": duration, "tokens": usage["total_tokens"],
          "tools_called": tool_count,
          "message": f"{agent_label} complete ({duration}s, {usage['total_tokens']} tokens)"})

    return result, duration, usage


async def process_bom_streamed(file_path: str) -> dict:
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

    pipeline_start = time.time()

    with trace("BOM Pipeline") as t:
        pipeline_meta["trace_id"] = t.trace_id
        pipeline_meta["trace_url"] = f"https://platform.openai.com/traces/{t.trace_id}"

        # ── Vision pre-processing ──
        if os.path.isfile(file_path) and (_is_image(file_path) or _is_pdf(file_path)):
            pipeline_meta["vision_used"] = True
            emit({"event": "agent_start", "agent": 0, "name": "Vision Extractor",
                  "message": "Extracting BOM from image with GPT-4.1 Vision..."})

            t0 = time.time()
            vision_input = _build_vision_input(file_path)
            vision_result = await Runner.run(bom_vision_agent, input=vision_input)
            bom_content = vision_result.final_output
            dur = round(time.time() - t0, 1)
            usage = _extract_usage(vision_result)

            pipeline_meta["trace_spans"].append({
                "agent": "BOM_Vision_Extractor", "model": "gpt-4.1",
                "duration_s": dur, "usage": usage,
                "status": "error" if bom_content.startswith("ERROR:") else "ok",
            })

            emit({"event": "agent_complete", "agent": 0, "name": "Vision Extractor",
                  "duration_s": dur, "tokens": usage["total_tokens"],
                  "message": f"Extracted {bom_content.count(chr(10))} lines from image"})

            if bom_content.startswith("ERROR:"):
                emit({"event": "error", "message": bom_content})
                return {"error": bom_content, "pipeline_meta": pipeline_meta}

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
            # ── Stage 1: BOM Parser & Product Matcher ──
            emit({"event": "guardrail", "type": "input", "status": "checking",
                  "message": "Validating BOM format..."})
            try:
                result1, dur1, usage1 = await run_agent_streamed(
                    bom_parser_and_matcher, input_msg, 1, "BOM Parser & Product Matcher")
                pipeline_meta["guardrails"]["input"]["checks"] = ["format", "content"]
                pipeline_meta["guardrails"]["input"]["status"] = "passed"
                emit({"event": "guardrail", "type": "input", "status": "passed",
                      "message": "Input guardrail passed"})
                pipeline_meta["trace_spans"].append({
                    "agent": "BOM_Parser_Product_Matcher", "model": "gpt-4.1",
                    "duration_s": dur1, "usage": usage1,
                    "tools": ["semanticSearch", "getProductDetails", "findAlternatives"],
                    "status": "ok",
                })
            except InputGuardrailTripwireTriggered as e:
                info = e.guardrail_result.output.output_info
                pipeline_meta["guardrails"]["input"] = {
                    "status": "blocked",
                    "reasons": info.get("reasons", ["Input validation failed"]),
                }
                emit({"event": "guardrail", "type": "input", "status": "blocked",
                      "message": "; ".join(info.get("reasons", []))})
                emit({"event": "error", "message": "Input guardrail blocked"})
                return {
                    "error": "Input guardrail blocked: " + "; ".join(info.get("reasons", [])),
                    "pipeline_meta": pipeline_meta,
                }

            # ── Stage 2: Datasheet Specialist ──
            result2, dur2, usage2 = await run_agent_streamed(
                datasheet_specialist,
                f"Enrich these matched products with technical specs from datasheets:\n\n{result1.final_output}",
                2, "Datasheet Specialist")
            pipeline_meta["trace_spans"].append({
                "agent": "Datasheet_Specialist", "model": "gpt-4.1",
                "duration_s": dur2, "usage": usage2,
                "tools": ["file_search (65 datasheets)"],
                "status": "ok",
            })

            # ── Stage 3: Quote Generator ──
            emit({"event": "guardrail", "type": "output", "status": "checking",
                  "message": "Will validate quote completeness..."})
            try:
                result3, dur3, usage3 = await run_agent_streamed(
                    quote_generator,
                    f"Generate a quote for these enriched products:\n\n{result2.final_output}",
                    3, "Quote Generator")
                pipeline_meta["guardrails"]["output"]["checks"] = [
                    "json_valid", "has_items", "has_financials", "pricing_reasonable"
                ]
                pipeline_meta["guardrails"]["output"]["status"] = "passed"
                emit({"event": "guardrail", "type": "output", "status": "passed",
                      "message": "Output guardrail passed"})
                pipeline_meta["trace_spans"].append({
                    "agent": "Quote_Generator", "model": "gpt-4.1",
                    "duration_s": dur3, "usage": usage3,
                    "tools": ["generateQuote"],
                    "status": "ok",
                })
            except OutputGuardrailTripwireTriggered as e:
                info = e.guardrail_result.output.output_info
                pipeline_meta["guardrails"]["output"] = {
                    "status": "warning",
                    "reasons": info.get("reasons", ["Output validation failed"]),
                }
                emit({"event": "guardrail", "type": "output", "status": "warning",
                      "message": "; ".join(info.get("reasons", []))})
                return {
                    "warning": "Output guardrail flagged",
                    "raw_output": result3.final_output if hasattr(result3, "final_output") else str(e),
                    "pipeline_meta": pipeline_meta,
                }

    pipeline_meta["total_duration_s"] = round(time.time() - pipeline_start, 1)
    total_input = sum(s.get("usage", {}).get("input_tokens", 0) for s in pipeline_meta["trace_spans"])
    total_output = sum(s.get("usage", {}).get("output_tokens", 0) for s in pipeline_meta["trace_spans"])
    pipeline_meta["total_tokens"] = total_input + total_output
    pipeline_meta["input_tokens"] = total_input
    pipeline_meta["output_tokens"] = total_output
    cost = (total_input / 1_000_000) * 2.00 + (total_output / 1_000_000) * 8.00
    pipeline_meta["estimated_cost_usd"] = round(cost, 4)

    emit({"event": "done",
          "message": f"Pipeline complete. {pipeline_meta['total_duration_s']}s, "
                     f"{pipeline_meta['total_tokens']} tokens, ${pipeline_meta['estimated_cost_usd']:.4f}"})

    try:
        result = json.loads(result3.final_output)
    except (json.JSONDecodeError, TypeError):
        result = {"status": "COMPLETED", "raw_output": result3.final_output}

    result["pipeline_meta"] = pipeline_meta
    return result


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python3 openai_bom_pipeline_stream.py <file_path>"}))
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print(json.dumps({"error": "OPENAI_API_KEY not set"}))
        sys.exit(1)

    result = await process_bom_streamed(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
