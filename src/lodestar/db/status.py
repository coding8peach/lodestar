"""Job lifecycle statuses and the allowed moves between them (see CLAUDE.md).

v0 uses: queued -> analyzed -> awaiting_review -> approved / rejected.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Literal, get_args

JobStatus = Literal[
    "discovered", "deduplicated", "queued", "analyzed", "awaiting_review",
    "approved", "rejected", "resume_tailored", "ready_to_apply", "applied",
]
STATUSES: tuple[str, ...] = get_args(JobStatus)

# new status -> statuses a job may move from
ALLOWED_FROM: dict[str, tuple[str, ...]] = {
    "deduplicated": ("discovered",),
    "queued": ("deduplicated",),
    "analyzed": ("queued",),
    "awaiting_review": ("analyzed",),
    "approved": ("awaiting_review",),
    "rejected": ("awaiting_review", "queued"),  # queued -> rejected: dismissed without analysis
    "resume_tailored": ("approved",),
    "ready_to_apply": ("resume_tailored",),
    "applied": ("ready_to_apply",),
}


class InvalidTransition(Exception):
    """A status change that the lifecycle doesn't allow, or on a missing job."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def transition(conn: sqlite3.Connection, job_id: str, new: JobStatus) -> None:
    """Move a job to `new` if its current status allows it. Does not commit.

    The check and the change are one conditional UPDATE, so two processes
    can't both make the same move.
    """
    allowed = ALLOWED_FROM.get(new)
    if not allowed:
        raise InvalidTransition(f"no job can move to {new!r}")
    placeholders = ",".join("?" * len(allowed))
    cur = conn.execute(
        f"UPDATE jobs SET status = ?, updated_at = ? WHERE id = ? AND status IN ({placeholders})",
        (new, now_utc(), job_id, *allowed),
    )
    if cur.rowcount == 1:
        return
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise InvalidTransition(f"job {job_id} not found")
    raise InvalidTransition(f"job {job_id}: can't move from {row[0]!r} to {new!r}")
