"""How clients start the Lodestar MCP server as a stdio subprocess.

Shared by every client (the fit agent's toolset, the LangGraph ingest node) so they
all launch it the same way. This is the server's launch recipe, not its internals.
"""

import os
import sys
from typing import TextIO

from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from lodestar.paths import DATA_DIR, PROJECT_ROOT


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "lodestar.mcp_server.server"],
        cwd=str(PROJECT_ROOT),
        # MCP passes only a minimal environment to subprocesses; forward our own settings
        # (LODESTAR_DB, LODESTAR_PROFILE) so the server reads the same data as its client.
        env=get_default_environment() | {k: v for k, v in os.environ.items() if k.startswith("LODESTAR_")},
    )


def server_errlog() -> TextIO:
    """Where the server subprocess's stderr (its logs) goes: data/logs/mcp_server.log.

    Keeps the client's terminal readable while keeping the server's logs for debugging.
    Opened in append mode and left open for the life of the client process.
    """
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return open(log_dir / "mcp_server.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
