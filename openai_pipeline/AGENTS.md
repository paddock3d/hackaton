# AGENTS.md -- Smart BOM Procurement Agent

## Overview

AI-powered BOM (Bill of Materials) processing pipeline using OpenAI Agents SDK. Converts raw BOM spreadsheets into optimized procurement quotes through 4 specialized agents with handoffs.

## Architecture

```
User uploads BOM CSV via SAP Fiori
    → SAP CAP backend uploads to S3, calls Python pipeline
    → Orchestrator Agent hands off sequentially:
        1. BOM_Parser_Product_Matcher (parse CSV, match against catalog)
        2. Datasheet_Specialist (enrich with technical specs via File Search)
        3. Quote_Generator (volume discounts, cross-sell, official quote)
    → Result returns to Fiori for approval/display
```

## Dev environment tips

- Python 3.12+ with venv: `.venv/bin/python3`
- Install deps: `pip install -r requirements.txt`
- SAP CAP backend must be running on port 4004 (SQLite mode: `cds watch` from `/opt/agentic/autoparts-inventory`)
- MCP server must be running on port 8100 (`python3 mcp_server.py` from `/opt/agentic/smart-part-finder-mcp`)
- Set `OPENAI_API_KEY` env var before running

## Setup

1. Install: `pip install -r requirements.txt`
2. Upload datasheets: `python3 setup_vector_store.py` (one-time, creates `.vector_store_id`)
3. Start SAP CAP: `cd /opt/agentic/autoparts-inventory && cds watch`
4. Start MCP: `cd /opt/agentic/smart-part-finder-mcp && python3 mcp_server.py`

## Testing

```bash
export OPENAI_API_KEY=sk-proj-...

# Test single agent (BOM Parser & Matcher only)
python3 test_pipeline.py single

# Test full 3-agent pipeline with handoffs
python3 test_pipeline.py full

# Process a real BOM from S3
python3 openai_bom_pipeline.py bom-uploads/REQ-123/bom.csv
```

## Key files

| File | Purpose |
|------|---------|
| `agents_def.py` | Agent definitions (4 agents + orchestrator with handoffs) |
| `tools.py` | MCP server connections (catalog port 8100, S3 port 8102) |
| `openai_bom_pipeline.py` | CLI entry point for SAP CAP to call |
| `test_pipeline.py` | Test harness (single agent or full pipeline) |
| `setup_vector_store.py` | One-time: upload 66 datasheets to OpenAI vector store |
| `agent_instructions/` | Instruction files for each agent |

## OpenAI technologies used

1. **OpenAI Agents SDK** -- 4 agents with typed handoffs
2. **OpenAI function calling** -- 11 tools via native MCP integration
3. **OpenAI File Search** -- 66 datasheets in vector store for RAG
4. **OpenAI GPT-4.1** -- model for all agents
5. **OpenAI tracing** -- built-in observability for agent decisions
6. **OpenAI structured outputs** -- every agent returns typed JSON

## Coding conventions

- Python, async/await throughout
- No comments unless the WHY is non-obvious
- Agent instructions live in `agent_instructions/` as plain text files
- MCP servers are connected via the SDK's native `MCPServerSse` (not manual HTTP)
- Never hardcode the API key in code files
