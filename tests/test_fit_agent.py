"""Fit agent tests with scripted fake models: no real LLM calls."""

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from lodestar.db import connect, init_db, save_job
from lodestar.fit_agent.agent import (
    EXAMPLE_OUTPUT,
    FIT_TOOLS,
    INSTRUCTION,
    PROMPT_VERSION,
    build_fit_agent,
    compact_tool_result,
)
from lodestar.fit_agent.models import (
    ThoughtlessLiteLlm,
    is_fallback_error,
    make_llm,
    model_names_from_env,
    strip_thoughts,
    validate_model_names,
)
from lodestar.fit_agent.runner import AllModelsFailed, analyze_job, describe_error
from lodestar.schemas import FitAnalysis, JobPosting, job_id_from_url

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"
URL = "https://boards.greenhouse.io/acme/jobs/1"
JOB_ID = job_id_from_url(URL)


from fakes import FakeLlm, RateLimited, text, tool_calls  # noqa: E402


TOOL_CALLS = tool_calls(JOB_ID)
ANSWER = json.dumps({**EXAMPLE_OUTPUT, "job_id": JOB_ID})


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A temp database with one job and the example profile, visible to the MCP subprocess."""
    db, profile = tmp_path / "t.sqlite", tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, profile)
    monkeypatch.setenv("LODESTAR_DB", str(db))
    monkeypatch.setenv("LODESTAR_PROFILE", str(profile))
    job = JobPosting(id=JOB_ID, url=URL, source="pasted_url", classified_by="url_rule",
                     title="Backend Engineer", company="Acme", description="Python and PostgreSQL. " * 20,
                     fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc))
    conn = connect(db)
    init_db(conn)
    save_job(conn, job)
    conn.close()


def run(models: dict[str, FakeLlm]):
    return asyncio.run(analyze_job(JOB_ID, list(models), make_model=lambda name: models[name]))


# --- per-run fallback, with the real MCP server as a subprocess -------------------

def test_single_model_run(data):
    fake = FakeLlm(model="a", script=[TOOL_CALLS, text(ANSWER)])
    result = run({"a": fake})
    assert result.model == "a" and result.skipped == []
    assert result.analysis.job_id == JOB_ID
    # the second request carries the tool results, compacted
    tool_msgs = [p.function_response for c in fake.requests[1].contents for p in c.parts if p.function_response]
    by_name = {fr.name: fr.response for fr in tool_msgs}
    assert by_name["get_job"]["title"] == "Backend Engineer"
    assert "content" not in by_name["get_job"] and "fetched_at" not in by_name["get_job"]
    assert by_name["get_profile"]["name"] == "Jane Example"


def test_mid_run_failure_starts_over_on_next_model(data):
    a = FakeLlm(model="a", script=[TOOL_CALLS, RateLimited("too large")])  # fails on turn 2
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)])
    result = run({"a": a, "b": b})
    assert result.model == "b"
    assert result.skipped and result.skipped[0].startswith("a: RateLimited")
    # b started from scratch: its first request has only the user's message, none of a's tool calls
    first = b.requests[0].contents
    assert len(first) == 1 and first[0].role == "user"


def test_non_fallback_error_is_raised(data):
    a = FakeLlm(model="a", script=[ValueError("bad request")])
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)])
    with pytest.raises(ValueError, match="bad request"):
        run({"a": a, "b": b})
    assert b.requests == []


def test_all_models_fail(data):
    a = FakeLlm(model="a", script=[RateLimited()])
    b = FakeLlm(model="b", script=[RateLimited()])
    with pytest.raises(AllModelsFailed, match="every model failed"):
        run({"a": a, "b": b})


def test_bad_json_retried_on_same_model(data):
    a = FakeLlm(model="a", script=[TOOL_CALLS, text("sorry, here it is"), text(ANSWER)])
    result = run({"a": a})
    assert result.model == "a" and len(a.requests) == 3


# --- models ------------------------------------------------------------------------

def test_model_routing():
    assert isinstance(make_llm("gemini-3.1-flash-lite"), Gemini)
    assert isinstance(make_llm("gemini/gemini-3.1-flash-lite"), Gemini)
    assert isinstance(make_llm("groq/openai/gpt-oss-120b"), ThoughtlessLiteLlm)


def test_models_from_env(monkeypatch):
    monkeypatch.setenv("LODESTAR_FIT_MODELS", " gemini-x , groq/openai/gpt-oss-120b ,")
    assert model_names_from_env() == ["gemini-x", "groq/openai/gpt-oss-120b"]
    monkeypatch.setenv("LODESTAR_FIT_MODELS", "gemini-x,<gemini model id>")
    with pytest.raises(ValueError, match="placeholders"):
        model_names_from_env()
    monkeypatch.delenv("LODESTAR_FIT_MODELS")
    with pytest.raises(ValueError, match="LODESTAR_FIT_MODELS"):
        model_names_from_env()
    assert validate_model_names(["openai/gpt-5.4-nano"]) == ["openai/gpt-5.4-nano"]


def test_real_provider_errors_recognized():
    import litellm
    from google.genai import errors

    assert is_fallback_error(litellm.RateLimitError("slow down", "groq", "m"))
    assert is_fallback_error(errors.ClientError(429, {"error": {"message": "quota"}}))
    assert not is_fallback_error(errors.ClientError(400, {"error": {"message": "bad"}}))
    assert not is_fallback_error(ValueError("x"))
    wrapped = RuntimeError("node failed")
    wrapped.__cause__ = litellm.RateLimitError("slow down", "groq", "m")
    assert is_fallback_error(wrapped)


def test_strip_thoughts():
    request = LlmRequest(contents=[
        types.Content(role="user", parts=[types.Part(text="Analyze job_1")]),
        types.Content(role="model", parts=[
            types.Part(text="I should fetch the job first.", thought=True),
            types.Part(function_call=types.FunctionCall(name="get_job", args={"job_id": "job_1"})),
        ]),
        types.Content(role="model", parts=[types.Part(text="only thinking", thought=True)]),
    ])
    strip_thoughts(request)
    assert len(request.contents) == 2
    assert [p.function_call.name for p in request.contents[1].parts] == ["get_job"]


# --- agent setup -------------------------------------------------------------------

def test_agent_setup():
    agent = build_fit_agent(FakeLlm(model="fake", script=[]))
    assert agent.tools[0].tool_filter == FIT_TOOLS == ["get_job", "get_profile"]
    assert "never produce a score" in INSTRUCTION
    FitAnalysis.model_validate({**EXAMPLE_OUTPUT, "job_id": "job_x"})  # the example matches the schema


@pytest.mark.parametrize("rule", [
    "One entry per distinct ask; merge near-duplicates",
    "background checks",                          # hiring conditions are not requirements
    "job title or seniority level alone is not evidence",
    "backend languages are related to each other",
    "cloud and hosting platforms are related",
    'Set "hard_constraint": true',                # hard constraints are marked by the agent
    "against the profile's targets",
    "a degree does not show location",            # evidence must support the requirement
])
def test_prompt_v2c_rules_present(rule):
    assert PROMPT_VERSION == "v2c"
    assert rule in INSTRUCTION


def test_example_marks_a_hard_constraint():
    assert [r["hard_constraint"] for r in EXAMPLE_OUTPUT["requirements"]] == [False, False, True]


def test_v2b_drops_the_split_rule():
    # v2's "make a separate entry for each" inflated requirement counts (30 -> 41 per run)
    assert "make a separate entry" not in INSTRUCTION


def test_compact_tool_result():
    response = {
        "content": [{"type": "text", "text": "{...same data, pretty-printed...}"}],
        "structuredContent": {"id": "job_1", "title": "Engineer", "url": "https://x", "company": None,
                              "fetched_at": "2026-09-27", "status": "queued", "description": "Python"},
        "isError": False,
    }
    tool = SimpleNamespace(name="get_job")
    assert compact_tool_result(tool, {}, None, response) == {"id": "job_1", "title": "Engineer",
                                                              "description": "Python"}
    error = {"content": [{"type": "text", "text": "job not found"}], "isError": True}
    assert compact_tool_result(tool, {}, None, error) is None  # unchanged


def test_describe_error_finds_the_message():
    class Wrapper(Exception):
        pass

    class ResourceExhaustedError(Exception):
        code = 429

    try:
        try:
            raise ResourceExhaustedError("429 RESOURCE_EXHAUSTED: quota exceeded for requests per day")
        except ResourceExhaustedError as inner:
            raise Wrapper("") from inner
    except Wrapper as e:
        text = describe_error(e)
    assert "HTTP 429" in text and "quota or rate limit" in text and "requests per day" in text


# --- usage tracking -------------------------------------------------------------------

def test_usage_recorded_for_each_call(data):
    fake = FakeLlm(model="gemini-a", script=[TOOL_CALLS, text(ANSWER)], usage=(1000, 200, 50))
    result = run({"gemini-a": fake})
    assert [c.status for c in result.calls] == ["ok", "ok"]
    first = result.calls[0]
    assert (first.model, first.attempt, first.input_tokens) == ("gemini-a", 1, 1000)
    assert first.output_tokens == 250 and first.thinking_tokens == 50  # Gemini: thinking billed as output
    assert first.latency_ms is not None


def test_litellm_output_already_includes_thinking(data):
    fake = FakeLlm(model="a", script=[TOOL_CALLS, text(ANSWER)], usage=(1000, 200, 50))
    result = run({"a": fake})
    assert result.calls[0].output_tokens == 200 and result.calls[0].thinking_tokens == 50


def test_usage_includes_attempts_that_fell_through(data):
    a = FakeLlm(model="a", script=[TOOL_CALLS, RateLimited("too large")], usage=(900, 30, 0))
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)], usage=(800, 40, 0))
    result = run({"a": a, "b": b})
    assert [(c.model, c.attempt, c.status) for c in result.calls] == [
        ("a", 1, "ok"), ("a", 1, "rate_limited"), ("b", 2, "ok"), ("b", 2, "ok")]
    assert "HTTP 429" in result.calls[1].error


def test_all_models_failed_carries_the_calls(data):
    a = FakeLlm(model="a", script=[RateLimited()])
    b = FakeLlm(model="b", script=[RateLimited()])
    with pytest.raises(AllModelsFailed) as info:
        run({"a": a, "b": b})
    assert [(c.model, c.status) for c in info.value.calls] == [("a", "rate_limited"), ("b", "rate_limited")]


# --- retry on temporary overload ---------------------------------------------------------

class Overloaded(Exception):
    status_code = 503


def test_is_overloaded():
    from google.genai import errors

    from lodestar.fit_agent.models import is_overloaded

    assert is_overloaded(errors.ServerError(503, {"error": {"message": "This model is currently experiencing "
                                                             "high demand.", "status": "UNAVAILABLE"}}))
    assert is_overloaded(Overloaded("overloaded"))
    assert not is_overloaded(RateLimited("429 RESOURCE_EXHAUSTED quota"))
    assert not is_overloaded(ValueError("bad request"))


def test_overload_retries_the_same_model(data, monkeypatch):
    from lodestar.fit_agent import runner

    monkeypatch.setattr(runner, "OVERLOAD_RETRY_DELAYS", (0.0, 0.0))
    a = FakeLlm(model="a", script=[Overloaded("503 high demand"), TOOL_CALLS, text(ANSWER)])
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)])
    result = run({"a": a, "b": b})
    assert result.model == "a" and result.skipped == [] and b.requests == []
    assert [c.status for c in result.calls] == ["rate_limited", "ok", "ok"]  # the busy call is recorded


def test_overload_gives_up_after_the_retries(data, monkeypatch):
    from lodestar.fit_agent import runner

    monkeypatch.setattr(runner, "OVERLOAD_RETRY_DELAYS", (0.0, 0.0))
    a = FakeLlm(model="a", script=[Overloaded("503"), Overloaded("503"), Overloaded("503")])
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)])
    result = run({"a": a, "b": b})
    assert result.model == "b" and len(a.requests) == 3  # first try + 2 retries


def test_rate_limit_falls_back_without_retrying(data, monkeypatch):
    from lodestar.fit_agent import runner

    monkeypatch.setattr(runner, "OVERLOAD_RETRY_DELAYS", (0.0, 0.0))
    a = FakeLlm(model="a", script=[RateLimited("429")])
    b = FakeLlm(model="b", script=[TOOL_CALLS, text(ANSWER)])
    assert run({"a": a, "b": b}).model == "b" and len(a.requests) == 1
