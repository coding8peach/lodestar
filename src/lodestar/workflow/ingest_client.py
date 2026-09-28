"""Call the Lodestar MCP server's ingest_url tool, as a plain MCP client (no LLM)."""

from lodestar.mcp_server.client import ToolFailed, call_tool


class IngestFailed(RuntimeError):
    """The MCP server couldn't ingest the URL; the message is the server's."""


async def mcp_ingest(url: str) -> dict:
    """Start lodestar-mcp over stdio, call ingest_url, return its structured result:
    {job_id, status: "saved" | "already_saved", url, title, company}."""
    try:
        return await call_tool("ingest_url", {"url": url})
    except ToolFailed as e:
        raise IngestFailed(str(e)) from e
