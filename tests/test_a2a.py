"""The fit agent served over A2A, called through the real A2A client, in-process.

Fake models, the real MCP server subprocess, the real to_a2a app and a2a-sdk client
(over httpx's in-process ASGI transport, so no port is opened).
"""

import asyncio
import json
import shutil
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fakes import FakeLlm, RateLimited, text, tool_calls

from lodestar.db import connect, init_db, save_job
from lodestar.fit_agent.agent import EXAMPLE_OUTPUT
from lodestar.fit_agent.client import FitAgentError, FitAgentUnavailable, analyze_via_a2a
from lodestar.fit_agent.server import build_app
from lodestar.schemas import JobPosting, job_id_from_url

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"
URL = "https://boards.greenhouse.io/acme/jobs/1"
JOB_ID = job_id_from_url(URL)
BASE = "http://localhost:8001"
ANSWER = json.dumps({**EXAMPLE_OUTPUT, "job_id": JOB_ID})


@pytest.fixture
def data(tmp_path, monkeypatch):
    db, profile = tmp_path / "t.sqlite", tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, profile)
    monkeypatch.setenv("LODESTAR_DB", str(db))
    monkeypatch.setenv("LODESTAR_PROFILE", str(profile))
    with closing(connect(db)) as conn:
        init_db(conn)
        save_job(conn, JobPosting(id=JOB_ID, url=URL, source="pasted_url", classified_by="url_rule",
                                  title="Backend Engineer", company="Acme",
                                  description="Python and PostgreSQL. " * 20,
                                  fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc)))


def ask(models: dict[str, FakeLlm], message: str = JOB_ID, max_paid_usd: float | None = None):
    """Serve FitService with these fake models and send one message through the A2A client."""
    app = build_app(list(models), port=8001, make_model=lambda name: models[name])

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE, timeout=60) as http:
                return await analyze_via_a2a(message, base_url=BASE, httpx_client=http, max_paid_usd=max_paid_usd)

    return asyncio.run(go())


def test_agent_card():
    app = build_app(["m"], port=8001)

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as http:
                return (await http.get("/.well-known/agent-card.json")).json()

    card = asyncio.run(go())
    assert card["name"] == "fit_agent"
    assert "job_id" in card["description"]
    assert "8001" in json.dumps(card)  # advertises the port it was given


def test_round_trip(data):
    reply = ask({"a": FakeLlm(model="a", script=[tool_calls(JOB_ID), text(ANSWER)], usage=(500, 50, 0))})
    assert [(c.model, c.status, c.input_tokens) for c in reply.calls] == [("a", "ok", 500), ("a", "ok", 500)]
    assert reply.model == "a" and reply.skipped == []
    assert reply.analysis.job_id == JOB_ID
    assert len(reply.analysis.requirements) == len(EXAMPLE_OUTPUT["requirements"])


def test_fallback_happens_inside_the_service(data):
    a = FakeLlm(model="a", script=[tool_calls(JOB_ID), RateLimited("too large")])
    b = FakeLlm(model="b", script=[tool_calls(JOB_ID), text(ANSWER)])
    reply = ask({"a": a, "b": b})
    assert reply.model == "b"
    assert reply.skipped[0].startswith("a: RateLimited")


def test_unknown_job_is_a_failed_task(data):
    with pytest.raises(FitAgentError, match="not found"):
        ask({"a": FakeLlm(model="a", script=[])}, message="job_0000000000000000")


def test_message_without_job_id(data):
    with pytest.raises(FitAgentError, match="no job_id"):
        ask({"a": FakeLlm(model="a", script=[])}, message="please analyze something")


def test_all_models_failing_is_a_failed_task(data):
    with pytest.raises(FitAgentError, match="every model failed"):
        ask({"a": FakeLlm(model="a", script=[RateLimited()])})


def test_service_not_running():
    with pytest.raises(FitAgentUnavailable, match="lodestar-agent"):
        asyncio.run(analyze_via_a2a(JOB_ID, base_url="http://127.0.0.1:9"))



# --- paid-model budget -------------------------------------------------------------------

@pytest.fixture
def prices(monkeypatch):
    from lodestar import pricing
    monkeypatch.setattr(pricing, "_overrides", lambda: {
        "free-m": {"input_per_mtok": 1, "output_per_mtok": 1, "free_tier": True},
        "paid-m": {"input_per_mtok": 1, "output_per_mtok": 1}})


def test_spent_budget_keeps_paid_models_out(data, prices):
    free = FakeLlm(model="free-m", script=[RateLimited()])
    paid = FakeLlm(model="paid-m", script=[tool_calls(JOB_ID), text(ANSWER)])
    with pytest.raises(FitAgentError, match="every model failed"):
        ask({"free-m": free, "paid-m": paid}, max_paid_usd=0.0)
    assert paid.requests == []  # never tried


def test_remaining_budget_allows_paid_fallback(data, prices):
    free = FakeLlm(model="free-m", script=[RateLimited()])
    paid = FakeLlm(model="paid-m", script=[tool_calls(JOB_ID), text(ANSWER)])
    assert ask({"free-m": free, "paid-m": paid}, max_paid_usd=0.05).model == "paid-m"


def test_allowed_models(prices):
    from lodestar.fit_agent.service import allowed_models

    assert allowed_models(["free-m", "paid-m"], None) == ["free-m", "paid-m"]
    assert allowed_models(["free-m", "paid-m"], 0.0) == ["free-m"]
    assert allowed_models(["free-m", "mystery"], 0.0) == ["free-m"]  # unknown price counts as paid
    with pytest.raises(ValueError, match="no free model"):
        allowed_models(["paid-m"], 0.0)


def test_agent_status(monkeypatch):
    from lodestar.fit_agent.client import agent_status

    ok, message = agent_status("http://127.0.0.1:9", timeout=0.5)
    assert not ok and "lodestar-agent" in message
