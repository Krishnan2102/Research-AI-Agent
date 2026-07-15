"""
Shared helper for calling tools on rag_server.py from anywhere in the
agent codebase (main.py, app.py, etc). Centralizes the connect -> get
tools -> call by name -> parse result sequence so it isn't duplicated.
"""

import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

RAG_MCP_URL = "http://127.0.0.1:8001/sse"


def _parse_mcp_result(raw: Any) -> Any:
    """
    MCP tool results come back through langchain_mcp_adapters in one of
    a few shapes, depending on what the underlying tool returned:

    1. A plain string (some servers/tools return this directly) --
       JSON-decode it if possible.
    2. A list of content blocks, e.g.
       [{'type': 'text', 'text': '{"...": ...}', 'id': '...'}, ...]
       -- this is what you get for a tool that returns a Python list:
       each list item becomes its own text block, JSON-encoded
       individually. We need to JSON-decode each block's 'text' field
       and reassemble the original list.
    3. A single content block dict (same shape as above, but for a
       tool that returned one dict, not a list) -- decode its 'text'.
    4. Already-structured data (a plain dict/list with none of the
       'type'/'text' block wrapper) -- pass through untouched.

    This function normalizes all of the above back into the original
    Python object the tool function returned.
    """
    # Case 1: plain string
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    # Case 2: list of content blocks
    if isinstance(raw, list):
        decoded_items = []
        for block in raw:
            if isinstance(block, dict) and "text" in block:
                decoded_items.append(_parse_mcp_result(block["text"]))
            else:
                # Not a content block -- already real data, keep as-is.
                decoded_items.append(block)
        return decoded_items

    # Case 3: single content block dict
    if isinstance(raw, dict) and "text" in raw and "type" in raw:
        return _parse_mcp_result(raw["text"])

    # Case 4: already structured / nothing to do
    return raw


# Tools on rag_server.py whose Python return type is a single dict
# (Dict[str, Any]), not a list. MCP wraps every tool's return value as a
# list of content blocks regardless of whether the underlying value was
# a list or a single object, so we can't tell them apart generically --
# we have to know, per tool, which shape to expect. ingest_pdf returns
# one dict; retrieve_from_rag returns a list of dicts (even when that
# list happens to contain exactly one item).
_SINGLE_OBJECT_TOOLS = {"ingest_pdf"}


async def call_rag_tool(tool_name: str, **kwargs: Any) -> Any:
    """
    Connect to rag_server.py, call `tool_name` with kwargs, and return
    the parsed result.

    A fresh client is created per call rather than reused, matching the
    reconnect-per-query pattern already used for the Wikipedia MCP
    connection in main.py — MCP client connections are tied to the
    asyncio event loop that created them, so reusing one across separate
    asyncio.run() calls would fail.

    Requires rag_server.py to already be running (python rag_server.py).
    """
    client = MultiServerMCPClient(
        {
            "rag_remote": {
                "url": RAG_MCP_URL,
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}

    if tool_name not in tool_map:
        raise RuntimeError(
            f"Tool '{tool_name}' not found on the RAG MCP server. "
            f"Available tools: {list(tool_map.keys())}. "
            f"Is rag_server.py running at {RAG_MCP_URL}?"
        )

    raw_result = await tool_map[tool_name].ainvoke(kwargs)
    parsed = _parse_mcp_result(raw_result)

    if (
        tool_name in _SINGLE_OBJECT_TOOLS
        and isinstance(parsed, list)
        and len(parsed) == 1
    ):
        return parsed[0]

    return parsed