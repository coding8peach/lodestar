"""Reads and writes. Each save that changes a job's status does so in the same transaction."""

import json
import logging
import sqlite3
from typing import Literal

from lodestar.db.status import JobStatus, now_utc, transition
from lodestar.schemas.fit import FitAnalysis, FitResult, LlmCall, RequirementMatch
from lodestar.schemas.job import JobPosting

log = logging.getLogger(__name__)

_JOB_FIELDS = list(JobPosting.model_fields)


def save_job(conn: sqlite3.Connection, job: JobPosting) -> bool:
    """Insert a job with status 'queued'. Returns False if it was already saved (nothing changes)."""
    data = job.model_dump(mode="json")
    now = now_utc()
    columns = [*_JOB_FIELDS, "status", "created_at", "updated_at"]
    values = [data[f] for f in _JOB_FIELDS] + ["queued", now, now]
    with conn:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO jobs ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            values,
        )
    inserted = cur.rowcount == 1
    log.info("%s job %s", "saved" if inserted else "already had", job.id)
    return inserted


def get_job(conn: sqlite3.Connection, job_id: str) -> JobPosting | None:
    row = conn.execute(f"SELECT {', '.join(_JOB_FIELDS)} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return JobPosting.model_validate(dict(row)) if row else None


def get_status(conn: sqlite3.Connection, job_id: str) -> JobStatus | None:
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row["status"] if row else None


def set_status(conn: sqlite3.Connection, job_id: str, new: JobStatus) -> None:
    """Move a job along its lifecycle. Raises InvalidTransition if the move isn't allowed."""
    with conn:
        transition(conn, job_id, new)
    log.info("job %s -> %s", job_id, new)


def save_fit_result(
    conn: sqlite3.Connection, result: FitResult, model: str | None = None, run_id: int | None = None
) -> int:
    """Store a fit result and its requirement rows, and move the job to 'analyzed'. Returns the fit_result id."""
    with conn:
        transition(conn, result.job_id, "analyzed")
        cur = conn.execute(
            "INSERT INTO fit_results (job_id, overall_score, recommendation, summary, model, created_at, run_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (result.job_id, result.overall_score, result.recommendation, result.analysis.summary, model, now_utc(),
             run_id),
        )
        fit_result_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO requirement_matches"
            " (fit_result_id, position, requirement, priority, match_level, evidence, explanation, hard_constraint)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (fit_result_id, i, m.requirement, m.priority, m.match_level, json.dumps(m.evidence), m.explanation,
                 int(m.hard_constraint))
                for i, m in enumerate(result.analysis.requirements)
            ],
        )
    log.info("saved fit result %d for job %s (%s)", fit_result_id, result.job_id, result.recommendation)
    return fit_result_id


def get_latest_fit_result(conn: sqlite3.Connection, job_id: str) -> tuple[int, FitResult] | None:
    """The most recent fit result for a job, as (fit_result_id, FitResult)."""
    fit = conn.execute(
        "SELECT * FROM fit_results WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,)
    ).fetchone()
    if fit is None:
        return None
    rows = conn.execute(
        "SELECT * FROM requirement_matches WHERE fit_result_id = ? ORDER BY position", (fit["id"],)
    ).fetchall()
    matches = [
        RequirementMatch(
            requirement=r["requirement"], priority=r["priority"], evidence=json.loads(r["evidence"]),
            match_level=r["match_level"], explanation=r["explanation"],
            hard_constraint=bool(r["hard_constraint"]),
        )
        for r in rows
    ]
    result = FitResult(
        job_id=job_id,
        analysis=FitAnalysis(job_id=job_id, requirements=matches, summary=fit["summary"]),
        overall_score=fit["overall_score"],
        recommendation=fit["recommendation"],
    )
    return fit["id"], result


def get_fit_result_model(conn: sqlite3.Connection, fit_result_id: int) -> str | None:
    """Which LLM produced a stored fit result (None if unknown)."""
    row = conn.execute("SELECT model FROM fit_results WHERE id = ?", (fit_result_id,)).fetchone()
    return row["model"] if row else None


def save_decision(
    conn: sqlite3.Connection, fit_result_id: int, decision: Literal["approve", "reject"], note: str | None = None
) -> int:
    """Store the candidate's decision next to the analysis, and move the job to approved/rejected."""
    row = conn.execute("SELECT job_id FROM fit_results WHERE id = ?", (fit_result_id,)).fetchone()
    if row is None:
        raise ValueError(f"fit result {fit_result_id} not found")
    new_status = "approved" if decision == "approve" else "rejected"
    with conn:
        transition(conn, row["job_id"], new_status)
        cur = conn.execute(
            "INSERT INTO decisions (fit_result_id, decision, note, created_at) VALUES (?, ?, ?, ?)",
            (fit_result_id, decision, note, now_utc()),
        )
    log.info("decision %s on fit result %d (job %s)", decision, fit_result_id, row["job_id"])
    return cur.lastrowid


def get_decision(conn: sqlite3.Connection, fit_result_id: int) -> dict | None:
    """The latest decision on a fit result: {decision, note, created_at}, or None."""
    row = conn.execute(
        "SELECT decision, note, created_at FROM decisions WHERE fit_result_id = ? ORDER BY id DESC LIMIT 1",
        (fit_result_id,),
    ).fetchone()
    return dict(row) if row else None



# --- usage tracking ----------------------------------------------------------------

def start_run(conn: sqlite3.Connection, kind: Literal["workflow", "eval", "batch"], label: str | None = None) -> int:
    with conn:
        cur = conn.execute("INSERT INTO runs (kind, label, started_at) VALUES (?, ?, ?)", (kind, label, now_utc()))
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, analyses_ok: int, analyses_failed: int) -> None:
    with conn:
        conn.execute("UPDATE runs SET finished_at = ?, analyses_ok = ?, analyses_failed = ? WHERE id = ?",
                     (now_utc(), analyses_ok, analyses_failed, run_id))


def save_llm_calls(
    conn: sqlite3.Connection,
    calls: list[LlmCall],
    run_id: int | None,
    job_id: str | None,
    fit_result_id: int | None = None,
) -> None:
    """Store model calls with their cost estimate, computed now (prices change later)."""
    from lodestar.pricing import estimate

    rows = []
    for c in calls:
        list_cost, cost, source = estimate(c)
        rows.append((run_id, job_id, fit_result_id, c.model, c.attempt, c.status, c.input_tokens, c.output_tokens,
                     c.thinking_tokens, c.latency_ms, c.error, list_cost, cost, source, c.started_at.isoformat()))
    with conn:
        conn.executemany(
            "INSERT INTO llm_calls (run_id, job_id, fit_result_id, model, attempt, status, input_tokens,"
            " output_tokens, thinking_tokens, latency_ms, error, list_cost_usd, cost_usd, price_source, started_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


_USAGE_TOTALS = """
    COUNT(c.id) AS calls,
    SUM(CASE WHEN c.status != 'ok' THEN 1 ELSE 0 END) AS failed_calls,
    COALESCE(SUM(c.input_tokens), 0) AS input_tokens,
    COALESCE(SUM(c.output_tokens), 0) AS output_tokens,
    SUM(c.list_cost_usd) AS list_cost_usd,
    SUM(c.cost_usd) AS cost_usd,
    SUM(CASE WHEN c.id IS NOT NULL AND c.list_cost_usd IS NULL THEN 1 ELSE 0 END) AS unpriced_calls
"""


def usage_by(conn: sqlite3.Connection, by: Literal["run", "day", "model"], limit: int = 20) -> list[dict]:
    """Usage totals, newest first. Unknown prices are counted in unpriced_calls, not as $0."""
    if by == "run":
        sql = f"""SELECT r.id AS run_id, r.kind, r.label, r.started_at, r.analyses_ok, r.analyses_failed,
                         {_USAGE_TOTALS}
                  FROM runs r LEFT JOIN llm_calls c ON c.run_id = r.id
                  GROUP BY r.id ORDER BY r.id DESC LIMIT ?"""
    elif by == "day":
        sql = f"""SELECT substr(c.started_at, 1, 10) AS day, {_USAGE_TOTALS}
                  FROM llm_calls c GROUP BY day ORDER BY day DESC LIMIT ?"""
    elif by == "model":
        sql = f"""SELECT c.model, {_USAGE_TOTALS}
                  FROM llm_calls c GROUP BY c.model ORDER BY cost_usd DESC, calls DESC LIMIT ?"""
    else:
        raise ValueError(f"unknown grouping {by!r}")
    return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]



def paid_spend_today(conn: sqlite3.Connection) -> float:
    """Actual spend (cost_usd, so free-tier calls count as 0) on model calls started today (UTC)."""
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE substr(started_at, 1, 10) = ?",
                       (now_utc()[:10],)).fetchone()
    return float(row[0])



def dismiss_job(conn: sqlite3.Connection, job_id: str, reason: str) -> None:
    """Reject a queued job without analyzing it (no LLM call, no fit result)."""
    with conn:
        transition(conn, job_id, "rejected")
        conn.execute("UPDATE jobs SET dismissed_reason = ? WHERE id = ?", (reason, job_id))
    log.info("dismissed %s: %s", job_id, reason)


def review_queue(conn: sqlite3.Connection) -> list[dict]:
    """Analyzed jobs waiting for a decision, best first: top_pick, then possible (by score), then blocked."""
    rows = conn.execute("""
        SELECT j.id AS job_id, j.title, j.company, j.status, f.id AS fit_result_id, f.overall_score,
               f.recommendation
        FROM jobs j
        JOIN fit_results f ON f.id = (SELECT MAX(id) FROM fit_results WHERE job_id = j.id)
        WHERE j.status IN ('analyzed', 'awaiting_review')
    """).fetchall()
    order = {"top_pick": 0, "possible": 1, "blocked": 2}
    return sorted((dict(r) for r in rows), key=lambda r: (order[r["recommendation"]], -r["overall_score"]))



class DecisionError(ValueError):
    """A decision that can't be recorded (no analysis yet, or already decided)."""


def record_decision(
    conn: sqlite3.Connection, job_id: str, decision: Literal["approve", "reject", "skip"], note: str | None = None
) -> int | None:
    """The one place a candidate's decision is recorded (workflow review node, CLI, UI).

    Uses the job's latest analysis. approve/reject: store the decision and move the job to
    approved/rejected (via awaiting_review). skip: the job stays waiting, nothing stored.
    Returns the decision id, or None for skip.
    """
    if decision not in ("approve", "reject", "skip"):
        raise DecisionError(f"unknown decision {decision!r}")
    status = get_status(conn, job_id)
    if status is None:
        raise DecisionError(f"job {job_id} not found")
    if status in ("approved", "rejected"):
        raise DecisionError(f"job {job_id} is already {status}")
    latest = get_latest_fit_result(conn, job_id)
    if latest is None or status == "queued":
        raise DecisionError(f"job {job_id} has no analysis yet")
    fit_id, _ = latest
    if status == "analyzed":
        set_status(conn, job_id, "awaiting_review")
    if decision == "skip":
        return None
    return save_decision(conn, fit_id, decision, note or None)
