"""Call one Lodestar MCP tool from Python, starting the server over stdio (no LLM involved).

Used by the workflow (ingest_url) and the fit agent service (checking a job exists).
"""

import re

from mcp import Client
from mcp.client.stdio import stdio_client

from lodestar.mcp_server.params import server_errlog, server_params


class ToolFailed(RuntimeError):
    """The tool reported an error; the message is the server's."""


async def call_tool(name: str, arguments: dict) -> dict:
    """Return the tool's structured result, or raise ToolFailed with the server's message."""
    with server_errlog() as errlog:
        async with Client(stdio_client(server_params(), errlog=errlog)) as client:
            result = await client.call_tool(name, arguments)
    if result.is_error:
        message = " ".join(getattr(c, "text", "") for c in result.content).strip()
        message = re.sub(r"^Error executing tool \w+:\s*", "", message)  # the SDK's wrapper prefix
        raise ToolFailed(message or f"{name} failed")
    return result.structured_content
