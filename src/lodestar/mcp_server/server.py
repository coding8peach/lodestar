"""Lodestar MCP server (stdio).

    uv run lodestar-mcp

Over stdio, stdout carries the MCP protocol, so all logging goes to stderr.
"""

import logging
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from lodestar.db import JobStatus, connect, get_status, init_db
from lodestar.db import get_job as db_get_job
from lodestar.ingest import ExtractionError, IngestResult
from lodestar.ingest import ingest_url as run_ingest
from lodestar.paths import PROJECT_ROOT, profile_path
from lodestar.schemas import JobPosting, Profile, load_profile

log = logging.getLogger(__name__)

mcp = MCPServer(
    "lodestar",
    instructions=(
        "Lodestar job-search data. Use ingest_url to add a job posting by URL, "
        "get_job to read a saved posting by job_id, and get_profile to read the candidate's profile."
    ),
)


class JobRecord(JobPosting):
    """A saved posting plus where it is in Lodestar's lifecycle."""

    status: JobStatus


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(connect()) as conn:
        init_db(conn)
        yield conn


@mcp.tool()
def ingest_url(url: str) -> IngestResult:
    """Add one job posting by URL and return its job_id.

    Works for Greenhouse, Lever and Ashby posting URLs, and for pages that include
    schema.org JobPosting data. A URL that is already saved returns the existing job
    (status "already_saved") without fetching it again.
    """
    try:
        with _db() as conn:
            return run_ingest(conn, url)
    except (ExtractionError, ValueError) as e:
        raise ToolError(str(e)) from e
    except httpx.HTTPError as e:
        raise ToolError(f"couldn't fetch the posting: {e}") from e


@mcp.tool()
def get_job(job_id: str) -> JobRecord:
    """Return a saved job posting (full description included) and its lifecycle status."""
    with _db() as conn:
        job = db_get_job(conn, job_id)
        if job is None:
            raise ToolError(f"job {job_id} not found")
        return JobRecord(**job.model_dump(), status=get_status(conn, job_id))


@mcp.tool()
def get_profile() -> Profile:
    """Return the candidate's profile: target preferences, experience, projects,
    education, skills (with depth, last-used year and evidence ids) and languages."""
    path = profile_path()
    try:
        return load_profile(path)
    except FileNotFoundError as e:
        raise ToolError(f"profile not found at {path}") from e


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("starting Lodestar MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
