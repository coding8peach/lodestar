"""SQLite layer: jobs, fit results, requirement matches, decisions."""

from lodestar.db.connection import SCHEMA_VERSION, connect, default_db_path, init_db
from lodestar.db.repo import (
    DecisionError,
    dismiss_job,
    get_decision,
    get_fit_result_model,
    get_job,
    get_latest_fit_result,
    get_latest_resume,
    get_status,
    paid_spend_today,
    record_decision,
    review_queue,
    save_decision,
    finish_run,
    save_fit_result,
    save_job,
    save_llm_calls,
    save_resume,
    set_status,
    start_run,
    usage_by,
)
from lodestar.db.status import ALLOWED_FROM, STATUSES, InvalidTransition, JobStatus

__all__ = [
    "ALLOWED_FROM", "InvalidTransition", "JobStatus", "SCHEMA_VERSION", "STATUSES", "connect",
    "default_db_path", "get_decision", "get_fit_result_model", "get_job", "get_latest_fit_result", "get_status", "init_db", "save_decision",
    "save_fit_result", "save_job", "set_status", "DecisionError", "dismiss_job", "finish_run", "get_latest_resume", "record_decision", "save_resume", "paid_spend_today", "review_queue", "save_llm_calls", "start_run", "usage_by",
]
