"""Service layer (what the UI calls), against a seeded temporary database."""

from contextlib import closing
from datetime import datetime, timezone

import pytest

from lodestar.app import service
from lodestar.db import DecisionError, connect

from conftest import EXAMPLE_PROFILE

def test_review_queue_order_and_fields(seeded):
    items = service.review_queue()
    assert [i.job_id for i in items] == [seeded["top"], seeded["possible"], seeded["blocked"]]
    assert items[0].recommendation == "top_pick" and items[0].score == 1.0
    assert items[2].blocked_by == ["Eligible to work in Canada"]
    assert items[0].url.startswith("https://") and items[0].location == "San Francisco"


def test_review_queue_filters(seeded):
    assert [i.job_id for i in service.review_queue(recommendations=["possible"])] == [seeded["possible"]]
    assert [i.job_id for i in service.review_queue(company="Gamma")] == [seeded["blocked"]]
    assert [i.job_id for i in service.review_queue(search="platform")] == [seeded["possible"]]


def test_job_detail(seeded):
    d = service.job_detail(seeded["top"])
    assert d.fit.recommendation == "top_pick" and d.model == "gemini-test" and d.decision is None
    q = service.job_detail(seeded["queued_senior"])
    assert q.fit is None and q.status == "queued"
    assert [s.job_id for s in q.similar] == [seeded["queued_staff"]]  # Senior/Staff near-duplicate
    with pytest.raises(KeyError):
        service.job_detail("job_0000000000000000")


def test_record_decision(seeded):
    service.record_decision(seeded["top"], "approve", "apply this week")
    d = service.job_detail(seeded["top"])
    assert d.status == "approved" and d.decision.decision == "approve" and d.decision.note == "apply this week"
    with pytest.raises(DecisionError, match="already approved"):
        service.record_decision(seeded["top"], "reject")
    with pytest.raises(DecisionError, match="no analysis"):
        service.record_decision(seeded["queued_other"], "approve")


def test_skip_keeps_the_job_waiting(seeded):
    service.record_decision(seeded["possible"], "skip")
    assert service.job_detail(seeded["possible"]).status == "awaiting_review"
    assert seeded["possible"] in [i.job_id for i in service.review_queue()]


def test_reject_blocked(seeded):
    result = service.reject_blocked([seeded["blocked"], seeded["top"]])
    assert result.done == 1 and result.not_done == {seeded["top"]: "not blocked"}
    d = service.job_detail(seeded["blocked"])
    assert d.status == "rejected" and d.decision.note == "blocked: Eligible to work in Canada"


def test_dismiss(seeded):
    result = service.dismiss([seeded["queued_other"], seeded["top"]], "not interested")
    assert result.done == 1 and seeded["top"] in result.not_done
    assert service.job_detail(seeded["queued_other"]).dismissed_reason == "not interested"


def test_ranked_queue_collapses_near_duplicates(seeded):
    items = service.ranked_queue()
    assert len(items) == 2  # the Senior/Staff Agents pair counts once
    agents = next(i for i in items if "Agents" in i.title)
    assert agents.similar_count == 1


def test_funnel(seeded):
    service.record_decision(seeded["top"], "approve")
    f = service.funnel()
    assert f.by_status == {"analyzed": 2, "approved": 1, "queued": 3}
    assert f.count("queued", "analyzed") == 5 and f.discovered == 0


def test_decisions_include_dismissed(seeded):
    service.record_decision(seeded["top"], "approve", "yes")
    service.dismiss([seeded["queued_other"]], "golden set")
    all_items = service.decisions()
    assert {(i.job_id, i.decision, i.dismissed) for i in all_items} == {
        (seeded["top"], "approve", False), (seeded["queued_other"], "reject", True)}
    assert [i.job_id for i in service.decisions("approve")] == [seeded["top"]]
    assert service.decisions("reject")[0].recommendation is None


def test_usage_rows(seeded):
    from lodestar.db import save_llm_calls, start_run
    from lodestar.schemas import LlmCall

    with closing(connect()) as conn:
        run_id = start_run(conn, "batch", "budget 3")
        save_llm_calls(conn, [LlmCall(model="gemini-3.1-flash-lite", attempt=1, status="ok", input_tokens=1000,
                                      output_tokens=100, started_at=datetime(2026, 9, 28, tzinfo=timezone.utc))],
                       run_id, seeded["top"])
    (row,) = service.usage("run")
    assert row.key.startswith(f"#{run_id} batch") and row.calls == 1 and row.input_tokens == 1000
    assert service.usage("model")[0].key == "gemini-3.1-flash-lite"


# --- Phase 2: pipeline actions -----------------------------------------------------------

def test_spending(seeded, monkeypatch):
    monkeypatch.setenv("LODESTAR_PAID_USD_PER_DAY", "0.10")
    s = service.spending()
    assert s.paid_limit_usd == 0.10 and s.paid_today_usd == 0.0


def test_analyze_next_runs_the_top_jobs_and_reports_progress(seeded):
    from lodestar.schemas import FitAgentReply, FitAnalysis, RequirementMatch

    async def fake_agent(job_id, max_paid_usd=None):
        return FitAgentReply(model="m", analysis=FitAnalysis(job_id=job_id, summary="s", requirements=[
            RequirementMatch(requirement="Python", priority="must_have", evidence=[], match_level="direct",
                             explanation="")]))

    seen = []
    batch = service.analyze_next(budget=2, on_progress=seen.append, analyze_fn=fake_agent)
    assert batch.ok == 2 and batch.failed == 0 and batch.stopped is None and batch.run_id
    assert [r.recommendation for r in batch.results] == ["top_pick", "top_pick"]
    assert [r.job_id for r in seen] == [r.job_id for r in batch.results]
    assert len({r.job_id for r in batch.results} & {seeded["queued_senior"], seeded["queued_staff"]}) <= 1


def test_analyze_next_stops_when_the_agent_is_down(seeded):
    from lodestar.fit_agent.client import FitAgentUnavailable

    async def down(job_id, max_paid_usd=None):
        raise FitAgentUnavailable("fit agent not reachable at http://localhost:8001")

    batch = service.analyze_next(budget=3, analyze_fn=down)
    assert batch.ok == 0 and batch.failed == 1 and "not reachable" in batch.stopped


def test_analyze_next_with_empty_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("LODESTAR_DB", str(tmp_path / "empty.sqlite"))
    monkeypatch.setenv("LODESTAR_PROFILE", str(EXAMPLE_PROFILE))
    assert service.analyze_next(3).results == []


def test_discover_summary_and_company_backfill(seeded, monkeypatch):
    import httpx
    from test_discover import transport

    from lodestar.ingest import discover as discover_module
    from lodestar.ingest.discover import Board

    monkeypatch.setattr(discover_module, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(discover_module, "PAUSE_BETWEEN_BOARDS", 0.0)
    monkeypatch.setattr(discover_module, "load_watchlist", lambda: [Board("Acme", "ashby", "acme")])
    # a job saved earlier without a company, which the board also lists
    url = "https://jobs.ashbyhq.com/acme/3f2a9c1e-1234-4abc-9def-0123456789a1"
    with closing(connect()) as conn:
        from lodestar.db import save_job
        from lodestar.schemas import JobPosting, job_id_from_url
        save_job(conn, JobPosting(id=job_id_from_url(url), url=url, source="pasted_url", classified_by="url_rule",
                                  title="Senior Backend Engineer", description="x" * 300,
                                  fetched_at=datetime(2026, 9, 1, tzinfo=timezone.utc)))
    with httpx.Client(transport=transport()) as client:
        summary = service.discover(client=client)
    (board,) = summary.boards
    assert board.listed == 7 and board.seen == 1 and board.queued == 1
    assert summary.skipped["title: ios"] == 1
    assert service.job_detail(job_id_from_url(url)).job.company == "Acme"  # filled in from the watchlist
