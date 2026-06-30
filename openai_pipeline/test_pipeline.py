#!/usr/bin/env python3
"""
Test the BOM Smart Processor pipeline.

Usage:
    export OPENAI_API_KEY=sk-proj-...
    python3 test_pipeline.py              # single agent test
    python3 test_pipeline.py full         # full pipeline test
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agents import Runner, trace
from agents_def import bom_parser_and_matcher, datasheet_specialist, quote_generator, orchestrator
from tools import catalog_mcp


SAMPLE_BOM_INPUT = """Process this BOM (already parsed, no S3 needed):

{
  "rfq_id": "BOM-TEST-001",
  "parsed_items": [
    {"line_number": 1, "part_number": "STM32F103C8T6", "description": "ARM Cortex-M3 MCU", "quantity": 100},
    {"line_number": 2, "part_number": "BME280", "description": "Humidity and pressure sensor", "quantity": 50},
    {"line_number": 3, "part_number": "LM7805", "description": "5V voltage regulator", "quantity": 200}
  ]
}

Match each part_number using semantic_search, then check availability and pricing, then find cross-sell recommendations.
Output the results as JSON with matched_items array."""


async def test_single_agent():
    """Test just the parser/matcher agent with a small BOM."""
    print("=" * 60)
    print("TEST: BOM Parser & Product Matcher (single agent)")
    print("=" * 60)

    async with catalog_mcp:
        result = await Runner.run(
            bom_parser_and_matcher,
            input=SAMPLE_BOM_INPUT,
        )
    print("\nResult:")
    print(result.final_output[:3000])
    return result.final_output


async def test_full_pipeline():
    """Test the full pipeline by chaining agents manually."""
    print("=" * 60)
    print("TEST: Full pipeline (sequential agent chain)")
    print("=" * 60)

    with trace("BOM Full Pipeline"):
        async with catalog_mcp:
            # Stage 1: Parse & Match
            print("\n[1/3] Running BOM Parser & Product Matcher...")
            result1 = await Runner.run(
                bom_parser_and_matcher,
                input=SAMPLE_BOM_INPUT,
            )
            matcher_output = result1.final_output
            print(f"  -> Matched items returned ({len(matcher_output)} chars)")

            # Stage 2: Datasheet Enrichment
            print("[2/3] Running Datasheet Specialist...")
            result2 = await Runner.run(
                datasheet_specialist,
                input=f"Enrich these matched products with technical specs from datasheets:\n\n{matcher_output}",
            )
            enriched_output = result2.final_output
            print(f"  -> Enriched output returned ({len(enriched_output)} chars)")

            # Stage 3: Quote Generation
            print("[3/3] Running Quote Generator...")
            result3 = await Runner.run(
                quote_generator,
                input=f"Generate a quote for these enriched products:\n\n{enriched_output}",
            )
            final_output = result3.final_output
            print(f"  -> Final quote returned ({len(final_output)} chars)")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(final_output[:5000])

    try:
        parsed = json.loads(final_output)
        print("\n" + "=" * 60)
        print("VALIDATION:")
        print("=" * 60)
        items = parsed.get("matched_items", parsed.get("line_items", []))
        print(f"  Line items: {len(items)}")
        fs = parsed.get("financial_summary", {})
        print(f"  Subtotal: ${fs.get('subtotal', 'N/A')}")
        print(f"  Discounts: ${fs.get('volume_discounts_applied', 'N/A')}")
        print(f"  Tax: ${fs.get('tax_amount', 'N/A')}")
        print(f"  Grand total: ${fs.get('grand_total_required', 'N/A')}")
        td = parsed.get("technical_documentation", {})
        if td:
            print(f"  Datasheets found: {td.get('items_with_datasheets', 'N/A')}")
            print(f"  Design notes: {td.get('total_design_notes', 'N/A')}")
        quote = parsed.get("quote_id", parsed.get("quote_details", {}).get("quote_id", "N/A"))
        print(f"  Quote ID: {quote}")
    except json.JSONDecodeError:
        print("\n  [WARNING] Output is not valid JSON")

    return final_output


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        await test_single_agent()
    elif mode == "full":
        await test_full_pipeline()
    else:
        print(f"Usage: python3 test_pipeline.py [single|full]")


if __name__ == "__main__":
    asyncio.run(main())
