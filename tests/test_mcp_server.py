"""Call the tools through a real MCP client session (in-memory transport)."""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mcp import Client

from lodestar.db import connect, init_db, save_job
from lodestar.ingest import service as service_module
from lodestar.mcp_server.server import mcp
from lodestar.schemas import JobPosting, job_id_from_url

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"
URL = "https://boards.greenhouse.io/acme/jobs/1"


def make_job(url: str = URL) -> JobPosting:
    return JobPosting(
        id=job_id_from_url(url), url=url, source="pasted_url", classified_by="url_rule",
        title="Senior Backend Engineer", company="Acme", description="Build agent tools in Python. " * 20,
        fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "t.sqlite"
    profile = tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, profile)
    monkeypatch.setenv("LODESTAR_DB", str(db))
    monkeypatch.setenv("LODESTAR_PROFILE", str(profile))
    return db, profile


def call(tool: str, args: dict):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(run())


def test_lists_three_tools():
    async def run():
        async with Client(mcp) as client:
            return await client.list_tools()
    names = {t.name for t in asyncio.run(run()).tools}
    assert names == {"ingest_url", "get_job", "get_fit_analysis", "get_profile"}


def test_get_job(env):
    db, _ = env
    with connect(db) as conn:
        init_db(conn)
        save_job(conn, make_job())
    result = call("get_job", {"job_id": job_id_from_url(URL)})
    assert not result.is_error
    assert result.structured_content["title"] == "Senior Backend Engineer"
    assert result.structured_content["status"] == "queued"


def test_get_job_missing(env):
    result = call("get_job", {"job_id": "job_0000000000000000"})
    assert result.is_error
    assert "not found" in result.content[0].text


def test_get_profile(env):
    result = call("get_profile", {})
    assert not result.is_error
    assert result.structured_content["targets"]["frontend_preference"] == "light_frontend_ok"


def test_get_profile_missing(env):
    _, profile = env
    profile.unlink()
    result = call("get_profile", {})
    assert result.is_error and "profile not found" in result.content[0].text


def test_ingest_url(env, monkeypatch):
    monkeypatch.setattr(service_module, "build_posting", lambda url, source="pasted_url": make_job(url))
    first = call("ingest_url", {"url": URL})
    assert first.structured_content["status"] == "saved"
    second = call("ingest_url", {"url": URL})
    assert second.structured_content["status"] == "already_saved"


def test_ingest_url_error_is_a_tool_error(env):
    result = call("ingest_url", {"url": "not a url"})
    assert result.is_error
    assert "not an http(s) URL" in result.content[0].text
