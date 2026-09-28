"""Workflow tests: the graph with fake ingest and fake fit agent, plus the real MCP ingest client."""

import asyncio
import shutil
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from langgraph.types import Command

from lodestar.db import connect, get_decision, get_status, init_db, save_job
from lodestar.fit_agent.client import FitAgentError, FitAgentUnavailable
from lodestar.schemas import FitAgentReply, FitAnalysis, JobPosting, RequirementMatch, job_id_from_url
from lodestar.workflow.graph import build_graph
from lodestar.workflow.ingest_client import IngestFailed, mcp_ingest

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"
URL = "https://boards.greenhouse.io/acme/jobs/1"
JOB_ID = job_id_from_url(URL)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "t.sqlite"
    profile = tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, profile)
    monkeypatch.setenv("LODESTAR_DB", str(path))
    monkeypatch.setenv("LODESTAR_PROFILE", str(profile))
    return path


def make_job() -> JobPosting:
    return JobPosting(id=JOB_ID, url=URL, source="pasted_url", classified_by="url_rule",
                      title="Backend Engineer", company="Acme", description="Python and PostgreSQL. " * 20,
                      fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc))


async def fake_ingest(url: str) -> dict:
    """Does what the MCP tool does: save the job, report saved / already_saved."""
    with closing(connect()) as conn:
        init_db(conn)
        saved = save_job(conn, make_job())
    return {"job_id": JOB_ID, "status": "saved" if saved else "already_saved", "url": URL,
            "title": "Backend Engineer", "company": "Acme"}


def make_run(job_id: str = JOB_ID) -> FitAgentReply:
    from lodestar.schemas import LlmCall
    calls = [LlmCall(model="gemini-test", attempt=1, status="ok", input_tokens=1000, output_tokens=100,
                     started_at=datetime(2026, 9, 28, tzinfo=timezone.utc))]
    analysis = FitAnalysis(job_id=job_id, summary="Good.", requirements=[
        RequirementMatch(requirement="Python", priority="must_have", evidence=["proj-quizbot"],
                         match_level="direct", explanation="QuizBot."),
        RequirementMatch(requirement="Kafka", priority="nice_to_have", evidence=[],
                         match_level="gap", explanation="None."),
    ])
    return FitAgentReply(analysis=analysis, model="gemini-test", tokens=123, calls=calls)


class FakeAgent:
    def __init__(self, error: Exception | None = None):
        self.calls, self.error = [], error

    async def __call__(self, job_id: str, max_paid_usd: float | None = None) -> FitAgentReply:
        self.calls.append(job_id)
        if self.error:
            raise self.error
        return make_run(job_id)


def run(agent, ingest=fake_ingest, url=URL, answer: dict | None = None) -> dict:
    """Run the graph; if it pauses for review and `answer` is given, resume with it."""
    async def go():
        graph = build_graph(ingest_fn=ingest, analyze_fn=agent)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = await graph.ainvoke({"url": url}, config)
        if state.get("__interrupt__") and answer is not None:
            state = await graph.ainvoke(Command(resume=answer), config)
        return state
    return asyncio.run(go())


def status() -> str:
    with closing(connect()) as conn:
        return get_status(conn, JOB_ID)


def test_happy_path(db):
    agent = FakeAgent()
    final = run(agent)
    assert "error" not in final
    assert final["ingest_status"] == "saved" and final["reused_analysis"] is False
    # must direct (2*1.0) + nice gap (1*0) = 2/3
    assert final["fit_result"]["overall_score"] == 0.667
    assert final["fit_result"]["recommendation"] == "possible"
    with closing(connect()) as conn:
        assert get_status(conn, JOB_ID) == "awaiting_review"  # paused at review (no answer given)
        row = conn.execute("SELECT model, overall_score FROM fit_results").fetchone()
        assert row["model"] == "gemini-test"
        assert conn.execute("SELECT COUNT(*) FROM requirement_matches").fetchone()[0] == 2
    assert agent.calls == [JOB_ID]


def test_same_url_reuses_analysis(db):
    run(FakeAgent())
    agent = FakeAgent()
    final = run(agent)
    assert final["ingest_status"] == "already_saved"
    assert final["reused_analysis"] is True and final["model"] == "gemini-test"
    assert agent.calls == []
    with closing(connect()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM fit_results").fetchone()[0] == 1


def test_ingest_error_ends_run(db):
    async def failing_ingest(url):
        raise IngestFailed("not an http(s) URL: 'x'")

    agent = FakeAgent()
    final = run(agent, ingest=failing_ingest)
    assert final["error"].startswith("ingest failed: not an http(s) URL")
    assert agent.calls == []


def test_all_models_fail_leaves_job_queued(db):
    final = run(FakeAgent(error=FitAgentError("every model failed: a: RateLimitError")))
    assert final["error"].startswith("analyze failed: every model failed")
    with closing(connect()) as conn:
        assert get_status(conn, JOB_ID) == "queued"
        assert conn.execute("SELECT COUNT(*) FROM fit_results").fetchone()[0] == 0
    # a later run retries the analysis
    assert "error" not in run(FakeAgent())


def test_agent_not_running_leaves_job_queued(db):
    final = run(FakeAgent(error=FitAgentUnavailable("fit agent not reachable at http://localhost:8001")))
    assert "not reachable" in final["error"]
    assert status() == "queued"


def test_graph_shape(db):
    graph = build_graph(ingest_fn=fake_ingest, analyze_fn=FakeAgent()).get_graph()
    assert set(graph.nodes) == {"__start__", "ingest", "analyze", "review", "__end__"}


# --- the real MCP ingest client, against the real server subprocess (no network) ---------

def test_mcp_ingest_reports_server_error(db):
    with pytest.raises(IngestFailed, match="not an http"):
        asyncio.run(mcp_ingest("not a url"))


def test_mcp_ingest_already_saved(db):
    with closing(connect(db)) as conn:
        init_db(conn)
        save_job(conn, make_job())
    out = asyncio.run(mcp_ingest(URL))
    assert out["status"] == "already_saved" and out["job_id"] == JOB_ID


# --- review ------------------------------------------------------------------------------

def test_review_pauses_with_ids(db):
    final = run(FakeAgent())
    (pause,) = final["__interrupt__"]
    assert pause.value == {"job_id": JOB_ID, "fit_result_id": final["fit_result_id"]}
    assert status() == "awaiting_review"


@pytest.mark.parametrize("decision, expected_status", [("approve", "approved"), ("reject", "rejected")])
def test_decision_is_stored_next_to_the_analysis(db, decision, expected_status):
    final = run(FakeAgent(), answer={"decision": decision, "note": "Mostly Go"})
    assert final["decision"] == decision and "error" not in final
    assert status() == expected_status
    with closing(connect()) as conn:
        stored = get_decision(conn, final["fit_result_id"])
    assert (stored["decision"], stored["note"]) == (decision, "Mostly Go")


def test_skip_leaves_job_awaiting_review(db):
    final = run(FakeAgent(), answer={"decision": "skip"})
    assert final["decision"] == "skip"
    assert status() == "awaiting_review"
    with closing(connect()) as conn:
        assert get_decision(conn, final["fit_result_id"]) is None


def test_awaiting_review_asks_again_without_reanalyzing(db):
    run(FakeAgent(), answer={"decision": "skip"})
    agent = FakeAgent()
    final = run(agent)
    assert agent.calls == [] and final["reused_analysis"] is True
    assert final["__interrupt__"]  # asks again
    final = run(agent, answer={"decision": "approve"})
    assert status() == "approved"


def test_decided_job_does_not_ask_again(db):
    run(FakeAgent(), answer={"decision": "reject", "note": "Too much frontend"})
    final = run(FakeAgent())
    assert "__interrupt__" not in final
    assert final["already_decided"] is True
    assert (final["decision"], final["note"]) == ("reject", "Too much frontend")
    assert status() == "rejected"


def test_unknown_decision_is_an_error(db):
    final = run(FakeAgent(), answer={"decision": "maybe"})
    assert final["error"].startswith("review failed")
    assert status() == "awaiting_review"


def test_usage_saved_with_the_fit_result(db):
    from lodestar.db import start_run

    with closing(connect()) as conn:
        init_db(conn)
        run_id = start_run(conn, "workflow", URL)

    async def go():
        graph = build_graph(ingest_fn=fake_ingest, analyze_fn=FakeAgent())
        return await graph.ainvoke({"url": URL, "run_id": run_id}, {"configurable": {"thread_id": "t"}})

    final = asyncio.run(go())
    with closing(connect()) as conn:
        row = conn.execute("SELECT run_id, job_id, fit_result_id, model, input_tokens FROM llm_calls").fetchone()
        assert tuple(row) == (run_id, JOB_ID, final["fit_result_id"], "gemini-test", 1000)
        assert conn.execute("SELECT run_id FROM fit_results").fetchone()[0] == run_id


def test_usage_command(db, capsys):
    from lodestar.workflow.cli import usage

    run(FakeAgent())  # no run_id: calls recorded without a run
    assert usage(["--by", "model"]) == 0
    out = capsys.readouterr().out
    assert "gemini-test" in out and "you pay" in out


# --- lodestar analyze (batch) -----------------------------------------------------------

def _queue(n: int):
    urls = [f"https://boards.greenhouse.io/acme/jobs/{900 + i}" for i in range(n)]
    with closing(connect()) as conn:
        init_db(conn)
        for i, url in enumerate(urls):
            save_job(conn, JobPosting(id=job_id_from_url(url), url=url, source="greenhouse",
                                      classified_by="url_rule", title=f"Backend Engineer {i}", company=f"Co{i}",
                                      description="Python and PostgreSQL. " * 20,
                                      fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc)))
    return [job_id_from_url(u) for u in urls]


def test_batch_analyzes_top_n_and_records_a_run(db, monkeypatch, capsys):
    from lodestar.workflow import analysis, cli

    _queue(4)
    seen = []
    real = analysis.analyze_one

    async def fake_analyze_one(job_id, run_id=None, analyze_fn=None, max_paid_usd=None):
        seen.append((job_id, run_id, max_paid_usd))
        return await real(job_id, run_id, FakeAgent(), max_paid_usd)

    monkeypatch.setattr(analysis, "analyze_one", fake_analyze_one)
    monkeypatch.setenv("LODESTAR_PAID_USD_PER_DAY", "0.05")
    assert cli.analyze_cmd(["--budget", "2"]) == 0
    assert len(seen) == 2 and all(m == 0.05 for _, _, m in seen)  # nothing spent yet: full budget left
    with closing(connect()) as conn:
        run = conn.execute("SELECT kind, analyses_ok, analyses_failed FROM runs").fetchone()
        assert tuple(run) == ("batch", 2, 0)
        assert conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'analyzed'").fetchone()[0] == 2
    assert "analyzed 2, failed 0" in capsys.readouterr().out


def test_batch_stops_when_the_agent_is_unreachable(db, monkeypatch, capsys):
    from lodestar.workflow import analysis, cli

    _queue(3)
    calls = []

    async def unreachable(job_id, run_id=None, analyze_fn=None, max_paid_usd=None):
        calls.append(job_id)
        return {"error": "analyze failed: fit agent not reachable", "error_kind": "unavailable"}

    monkeypatch.setattr(analysis, "analyze_one", unreachable)
    assert cli.analyze_cmd(["--budget", "3"]) == 1
    assert len(calls) == 1  # stopped after the first
    assert "stopping" in capsys.readouterr().out


def test_batch_dry_run_analyzes_nothing(db, monkeypatch, capsys):
    from lodestar.workflow import analysis, cli

    _queue(2)
    monkeypatch.setattr(analysis, "analyze_one", None)  # would fail if called
    assert cli.analyze_cmd(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 queued: 2 distinct roles" in out and "dry run" in out


# --- lodestar review ----------------------------------------------------------------------

def _analyzed_job(i: int, levels: list[str], hard_gap: bool = False) -> str:
    """Save a job with a stored analysis (status analyzed); returns its id."""
    from lodestar.db import save_fit_result
    from lodestar.scoring import score_analysis

    url = f"https://boards.greenhouse.io/acme/jobs/{700 + i}"
    jid = job_id_from_url(url)
    reqs = [RequirementMatch(requirement=f"r{k}", priority="must_have", evidence=[], match_level=lv,
                             explanation="") for k, lv in enumerate(levels)]
    if hard_gap:
        reqs.append(RequirementMatch(requirement="Eligible to work in Canada", priority="must_have", evidence=[],
                                     match_level="gap", explanation="", hard_constraint=True))
    with closing(connect()) as conn:
        init_db(conn)
        save_job(conn, JobPosting(id=jid, url=url, source="greenhouse", classified_by="url_rule",
                                  title=f"Engineer {i}", company=f"Co{i}", description="Python. " * 50,
                                  fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc)))
        save_fit_result(conn, score_analysis(FitAnalysis(job_id=jid, requirements=reqs, summary="s")), model="m")
    return jid


def _answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_graph_started_with_job_id_skips_ingest(db):
    jid = _analyzed_job(1, ["direct"])

    async def no_ingest(url):
        raise AssertionError("ingest must be skipped")

    async def go():
        graph = build_graph(ingest_fn=no_ingest, analyze_fn=FakeAgent())
        return await graph.ainvoke({"job_id": jid}, {"configurable": {"thread_id": "t"}})

    state = asyncio.run(go())
    assert state["reused_analysis"] is True and state["__interrupt__"]


def test_review_goes_best_first_and_bulk_rejects_blocked(db, monkeypatch, capsys):
    from lodestar.workflow import cli

    possible = _analyzed_job(1, ["related", "gap"])        # 0.3 possible
    top = _analyzed_job(2, ["direct", "direct"])           # 1.0 top_pick
    blocked = _analyzed_job(3, ["direct"], hard_gap=True)  # blocked
    _answers(monkeypatch, "a", "great fit", "r", "", "y")
    assert cli.review_cmd([]) == 0
    with closing(connect()) as conn:
        assert get_status(conn, top) == "approved"
        assert get_status(conn, possible) == "rejected"
        assert get_status(conn, blocked) == "rejected"
        notes = dict(conn.execute("SELECT f.job_id, d.note FROM decisions d JOIN fit_results f ON f.id = d.fit_result_id"))
    assert notes[top] == "great fit" and notes[blocked].startswith("blocked: Eligible to work in Canada")
    out = capsys.readouterr().out
    assert out.index("Engineer 2") < out.index("Engineer 1")  # top pick shown first
    assert "approved 1, rejected 2, skipped 0" in out


def test_review_quit_leaves_the_rest_waiting(db, monkeypatch, capsys):
    from lodestar.workflow import cli

    a = _analyzed_job(1, ["direct"])
    b = _analyzed_job(2, ["related"])
    _answers(monkeypatch, "q")
    assert cli.review_cmd([]) == 0
    with closing(connect()) as conn:
        assert get_status(conn, a) == "awaiting_review"   # shown, not decided
        assert get_status(conn, b) == "analyzed"          # not reached
    assert "2 still waiting for review" in capsys.readouterr().out


def test_review_limit(db, monkeypatch):
    from lodestar.workflow import cli

    _analyzed_job(1, ["direct"])
    second = _analyzed_job(2, ["related"])
    _answers(monkeypatch, "s")
    cli.review_cmd(["--limit", "1"])
    with closing(connect()) as conn:
        assert get_status(conn, second) == "analyzed"


def test_dismiss_queued_job(db, capsys):
    from lodestar.workflow import cli

    (jid,) = _queue(1)
    assert cli.review_cmd(["--dismiss", jid, "--reason", "golden set: Canada only"]) == 0
    with closing(connect()) as conn:
        row = conn.execute("SELECT status, dismissed_reason FROM jobs WHERE id = ?", (jid,)).fetchone()
        assert tuple(row) == ("rejected", "golden set: Canada only")
        assert conn.execute("SELECT COUNT(*) FROM fit_results").fetchone()[0] == 0  # no analysis needed


def test_dismiss_only_works_on_queued_jobs(db, capsys):
    from lodestar.workflow import cli

    jid = _analyzed_job(1, ["direct"])
    assert cli.review_cmd(["--dismiss", jid]) == 1
    assert "only queued jobs" in capsys.readouterr().out
