"""
MCP server connections for the BOM Smart Processor pipeline.

Uses the OpenAI Agents SDK's native MCP client support
instead of manual HTTP wrappers. The SDK handles SSE protocol,
tool discovery, and schema registration automatically.
"""
from agents.mcp import MCPServerSse, MCPServerSseParams

catalog_mcp = MCPServerSse(
    params=MCPServerSseParams(url="http://10.134.241.80:8100/sse"),
    name="catalog",
    cache_tools_list=True,
    client_session_timeout_seconds=30,
)

s3_mcp = MCPServerSse(
    params=MCPServerSseParams(url="http://10.134.241.80:8102/sse"),
    name="s3",
    cache_tools_list=True,
    client_session_timeout_seconds=30,
)

datasheet_mcp = MCPServerSse(
    params=MCPServerSseParams(url="http://10.134.241.80:8101/sse"),
    name="datasheets",
    cache_tools_list=True,
    client_session_timeout_seconds=30,
)

sap_mcp = MCPServerSse(
    params=MCPServerSseParams(url="http://10.14.82.214:31116/sse"),
    name="sap-s4hana",
    cache_tools_list=True,
    client_session_timeout_seconds=30,
)
