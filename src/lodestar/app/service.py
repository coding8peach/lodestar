"""Service functions: the only thing a UI calls. Each opens its own database connection."""

from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime
from typing import Literal

from lodestar.app.models import (
    AnalyzedJob,
    BatchResult,
    BoardSummary,
    BulkResult,
    Decision,
    DecisionItem,
    Funnel,
    JobDetail,
    QueueItem,
    ReviewItem,
    SimilarJob,
    Spending,
    DiscoverySummary,
    UsageRow,
)
from lodestar.db import (
    DecisionError,
    InvalidTransition,
    connect,
    dismiss_job,
    get_decision,
    get_fit_result_model,
    get_job,
    get_latest_fit_result,
    get_status,
    init_db,
    record_decision as _record_decision,
    review_queue as _review_queue,
    usage_by,
)
from lodestar.paths import profile_path
from lodestar.ranking import group_key, load_queue, load_rank_config, rank_queue
from lodestar.schemas import load_profile
from lodestar.scoring import unmet_hard_constraints


def _conn():
    conn = connect()
    init_db(conn)
    return closing(conn)


def _analyzed_at(conn, fit_result_id: int) -> datetime:
    row = conn.execute("SELECT created_at FROM fit_results WHERE id = ?", (fit_result_id,)).fetchone()
    return datetime.fromisoformat(row[0])


# --- review ---------------------------------------------------------------------------

def review_queue(
    recommendations: Iterable[str] | None = None, company: str | None = None, search: str | None = None
) -> list[ReviewItem]:
    """Analyzed jobs waiting for a decision: top_pick, then possible (by score), then blocked."""
    wanted = set(recommendations) if recommendations else None
    needle = (search or "").strip().lower()
    items = []
    with _conn() as conn:
        for row in _review_queue(conn):
            if wanted and row["recommendation"] not in wanted:
                continue
            if company and (row["company"] or "") != company:
                continue
            if needle and needle not in f"{row['title']} {row['company'] or ''}".lower():
                continue
            job = get_job(conn, row["job_id"])
            _, result = get_latest_fit_result(conn, row["job_id"])
            items.append(ReviewItem(
                job_id=row["job_id"], company=row["company"], title=row["title"], location=job.location,
                url=job.url, recommendation=row["recommendation"], score=row["overall_score"],
                blocked_by=[r.requirement for r in unmet_hard_constraints(result.analysis)],
                status=row["status"], analyzed_at=_analyzed_at(conn, row["fit_result_id"]),
            ))
    return items


def job_detail(job_id: str) -> JobDetail:
    with _conn() as conn:
        job = get_job(conn, job_id)
        if job is None:
            raise KeyError(f"job {job_id} not found")
        status = get_status(conn, job_id)
        dismissed = conn.execute("SELECT dismissed_reason FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
        detail = JobDetail(job=job, status=status, dismissed_reason=dismissed)
        latest = get_latest_fit_result(conn, job_id)
        if latest:
            fit_id, result = latest
            detail.fit, detail.fit_result_id = result, fit_id
            detail.model = get_fit_result_model(conn, fit_id)
            detail.analyzed_at = _analyzed_at(conn, fit_id)
            detail.blocked_by = [r.requirement for r in unmet_hard_constraints(result.analysis)]
            earlier = get_decision(conn, fit_id)
            if earlier:
                detail.decision = Decision(decision=earlier["decision"], note=earlier["note"],
                                           decided_at=datetime.fromisoformat(earlier["created_at"]))
        key = group_key(job)
        for row in conn.execute("SELECT id FROM jobs WHERE id != ? AND lower(company) IS lower(?)",
                                (job_id, job.company)):
            other = get_job(conn, row[0])
            if group_key(other) == key:
                detail.similar.append(SimilarJob(job_id=other.id, title=other.title,
                                                 status=get_status(conn, other.id)))
    return detail


def record_decision(job_id: str, decision: Literal["approve", "reject", "skip"], note: str | None = None) -> None:
    """Raises DecisionError when the job has no analysis or is already decided."""
    with _conn() as conn:
        _record_decision(conn, job_id, decision, note)


def reject_blocked(job_ids: Iterable[str]) -> BulkResult:
    """Reject blocked jobs, with the unmet hard constraint as the note."""
    result = BulkResult(done=0)
    for job_id in job_ids:
        detail = job_detail(job_id)
        if not detail.blocked_by:
            result.not_done[job_id] = "not blocked"
            continue
        try:
            record_decision(job_id, "reject", "blocked: " + "; ".join(detail.blocked_by))
            result.done += 1
        except DecisionError as e:
            result.not_done[job_id] = str(e)
    return result


def dismiss(job_ids: Iterable[str], reason: str) -> BulkResult:
    """Reject queued jobs without analyzing them."""
    result = BulkResult(done=0)
    with _conn() as conn:
        for job_id in job_ids:
            try:
                dismiss_job(conn, job_id, reason)
                result.done += 1
            except InvalidTransition as e:
                result.not_done[job_id] = f"only queued jobs can be dismissed ({e})"
    return result


# --- queue ------------------------------------------------------------------------------

def ranked_queue() -> list[QueueItem]:
    """Queued jobs in the order `lodestar analyze` would take them (near-duplicates collapsed)."""
    with _conn() as conn:
        queued, analyzed = load_queue(conn)
    ranked, _ = rank_queue(queued, analyzed, load_profile(profile_path()), load_rank_config())
    return [QueueItem(job_id=r.job.id, company=r.job.company, title=r.job.title, location=r.job.location,
                      url=r.job.url, score=r.score, matched=r.matched, similar_count=len(r.similar),
                      date_posted=r.job.date_posted) for r in ranked]


def funnel() -> Funnel:
    with _conn() as conn:
        by_status = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
        verdicts = Counter(dict(conn.execute("SELECT verdict, COUNT(*) FROM discoveries GROUP BY verdict").fetchall()))
    return Funnel(by_status=by_status, discovered=sum(verdicts.values()),
                  discovery_skipped=verdicts["skipped"] + verdicts["invalid"])


# --- decisions and usage --------------------------------------------------------------------

def decisions(decision: Literal["approve", "reject"] | None = None) -> list[DecisionItem]:
    """Decided jobs, newest first. Rejections include jobs dismissed without analysis."""
    items = []
    with _conn() as conn:
        rows = conn.execute("""
            SELECT d.decision, d.note, d.created_at, f.recommendation, f.overall_score, j.id, j.title, j.company, j.url
            FROM decisions d JOIN fit_results f ON f.id = d.fit_result_id JOIN jobs j ON j.id = f.job_id
        """).fetchall()
        for r in rows:
            items.append(DecisionItem(job_id=r["id"], company=r["company"], title=r["title"], url=r["url"],
                                      decision=r["decision"], note=r["note"],
                                      decided_at=datetime.fromisoformat(r["created_at"]),
                                      recommendation=r["recommendation"], score=r["overall_score"]))
        for r in conn.execute("SELECT id, title, company, url, dismissed_reason, updated_at FROM jobs "
                              "WHERE status = 'rejected' AND dismissed_reason IS NOT NULL").fetchall():
            items.append(DecisionItem(job_id=r["id"], company=r["company"], title=r["title"], url=r["url"],
                                      decision="reject", note=r["dismissed_reason"],
                                      decided_at=datetime.fromisoformat(r["updated_at"]),
                                      recommendation=None, score=None, dismissed=True))
    if decision:
        items = [i for i in items if i.decision == decision]
    return sorted(items, key=lambda i: i.decided_at, reverse=True)


def usage(by: Literal["run", "day", "model"] = "run", limit: int = 50) -> list[UsageRow]:
    with _conn() as conn:
        rows = usage_by(conn, by, limit)
    out = []
    for r in rows:
        key = f"#{r['run_id']} {r['kind']} {r['started_at'][:16]}" if by == "run" else str(r["day" if by == "day" else "model"])
        out.append(UsageRow(key=key, calls=r["calls"], failed_calls=r["failed_calls"] or 0,
                            input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                            list_cost_usd=r["list_cost_usd"], cost_usd=r["cost_usd"],
                            unpriced_calls=r["unpriced_calls"] or 0))
    return out



# --- pipeline actions (Phase 2) --------------------------------------------------------

def spending() -> Spending:
    import os

    from lodestar.db import paid_spend_today

    limit = float(os.environ.get("LODESTAR_PAID_USD_PER_DAY", "0") or 0)
    with _conn() as conn:
        return Spending(paid_limit_usd=limit, paid_today_usd=paid_spend_today(conn))


def discover(dry_run: bool = False, recheck: bool = False, client=None) -> DiscoverySummary:
    """List watched boards, dedupe, pre-filter, queue passing jobs. No LLM. Same code as `lodestar discover`."""
    from lodestar.ingest.discover import discover as run_discover
    from lodestar.ingest.discover import load_watchlist
    from lodestar.ingest.prefilter import load_rules

    with _conn() as conn:
        reports = run_discover(conn, load_watchlist(), load_rules(), dry_run=dry_run, recheck=recheck, client=client)
    return DiscoverySummary(
        boards=[BoardSummary(company=r.board.company, listed=r.listed, seen=r.seen, queued=len(r.queued),
                             skipped=dict(r.skipped), error=r.error) for r in reports],
        queued_jobs=[(j.company, j.title, j.location) for r in reports for j in r.queued],
        dry_run=dry_run,
    )


def analyze_next(budget: int = 3, on_progress=None, analyze_fn=None) -> BatchResult:
    """Analyze the top `budget` queued roles (ranking order), as a `batch` run.

    Stops early if the fit agent is unreachable or every model failed. Paid models are
    allowed only up to today's remaining LODESTAR_PAID_USD_PER_DAY. `on_progress(job)` is
    called after each job, for live display. Same code as `lodestar analyze`.
    """
    import asyncio

    from lodestar.db import finish_run, paid_spend_today, start_run
    from lodestar.ranking import pick
    from lodestar.workflow import analysis

    limit = spending().paid_limit_usd
    config = load_rank_config()
    with _conn() as conn:
        queued, analyzed = load_queue(conn)
    ranked, _ = rank_queue(queued, analyzed, load_profile(profile_path()), config)
    picked = pick(ranked, budget, config)
    if not picked:
        return BatchResult(run_id=None, paid_limit_usd=limit)

    with _conn() as conn:
        run_id = start_run(conn, "batch", f"budget {budget}")
    batch = BatchResult(run_id=run_id, paid_limit_usd=limit)

    async def go() -> None:
        for r in picked:
            with _conn() as conn:
                remaining = max(0.0, limit - paid_spend_today(conn))
            kwargs = {"analyze_fn": analyze_fn} if analyze_fn else {}
            out = await analysis.analyze_one(r.job.id, run_id, max_paid_usd=remaining, **kwargs)
            item = AnalyzedJob(job_id=r.job.id, company=r.job.company, title=r.job.title)
            if "error" in out:
                item.error = out["error"]
            else:
                item.recommendation = out["fit_result"]["recommendation"]
                item.score = out["fit_result"]["overall_score"]
            batch.results.append(item)
            if on_progress:
                on_progress(item)
            if out.get("error_kind") in ("unavailable", "all_models_failed"):
                batch.stopped = out["error"]
                return

    try:
        asyncio.run(go())
    finally:
        with _conn() as conn:
            finish_run(conn, run_id, batch.ok, batch.failed)
    return batch
