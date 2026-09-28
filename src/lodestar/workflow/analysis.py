"""Analyze one job: fit agent (over A2A) -> Python scoring -> save result and usage.

Shared by the workflow's analyze node (one pasted URL) and `lodestar analyze` (a batch
from the queue), so there is one path from a job id to a stored fit result.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import closing
from typing import Literal

from lodestar.db import (
    connect,
    get_fit_result_model,
    get_latest_fit_result,
    get_status,
    init_db,
    save_fit_result,
    save_llm_calls,
)
from lodestar.fit_agent.client import FitAgentError, FitAgentUnavailable, analyze_via_a2a
from lodestar.schemas.fit import FitAgentReply
from lodestar.scoring import score_analysis

log = logging.getLogger(__name__)

AnalyzeFn = Callable[..., Awaitable[FitAgentReply]]  # (job_id, max_paid_usd=None) -> reply
ErrorKind = Literal["unavailable", "all_models_failed", "other"]


def _error(e: Exception) -> tuple[str, ErrorKind]:
    first_line = (str(e).splitlines() or [""])[0][:300] or type(e).__name__
    if isinstance(e, FitAgentUnavailable):
        kind: ErrorKind = "unavailable"
    elif isinstance(e, FitAgentError) and "every model failed" in first_line:
        kind = "all_models_failed"
    else:
        kind = "other"
    return first_line, kind


async def analyze_one(
    job_id: str,
    run_id: int | None = None,
    analyze_fn: AnalyzeFn = analyze_via_a2a,
    max_paid_usd: float | None = None,
) -> dict:
    """Returns {fit_result_id, fit_result, model, reused_analysis} or {error, error_kind}.

    A job that already has an analysis reuses it (no LLM call). On failure the job stays
    queued, so a later run can retry.
    """
    with closing(connect()) as conn:
        init_db(conn)
        status = get_status(conn, job_id)
        if status is None:
            return {"error": f"analyze failed: job {job_id} is not in the database", "error_kind": "other"}
        if status != "queued":
            latest = get_latest_fit_result(conn, job_id)
            if latest is None:
                return {"error": f"analyze failed: job {job_id} is '{status}' but has no fit result",
                        "error_kind": "other"}
            fit_id, result = latest
            log.info("analyze: job already %s; reusing fit result %d", status, fit_id)
            return {"fit_result_id": fit_id, "fit_result": result.model_dump(mode="json"),
                    "model": get_fit_result_model(conn, fit_id), "reused_analysis": True}

    # Outside the DB connection: this takes a while and runs in another service.
    try:
        reply = await analyze_fn(job_id, max_paid_usd=max_paid_usd)
        if reply.analysis.job_id != job_id:
            raise FitAgentError(f"the fit agent answered for {reply.analysis.job_id}, not {job_id}")
        result = score_analysis(reply.analysis)
    except Exception as e:
        message, kind = _error(e)
        log.error("analyze failed: %s: %s", type(e).__name__, message)
        return {"error": f"analyze failed: {message}", "error_kind": kind}
    if not reply.calls:
        log.warning("the fit agent reported no model usage for a new analysis; "
                    "is `lodestar-agent` up to date? (restart it after updates)")
    with closing(connect()) as conn:
        fit_id = save_fit_result(conn, result, model=reply.model, run_id=run_id)
        save_llm_calls(conn, reply.calls, run_id, job_id, fit_id)
    log.info("analyze: %s (%.2f) by %s, saved as fit result %d",
             result.recommendation, result.overall_score, reply.model, fit_id)
    return {"fit_result_id": fit_id, "fit_result": result.model_dump(mode="json"),
            "model": reply.model, "reused_analysis": False}
