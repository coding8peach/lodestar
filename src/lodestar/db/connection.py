"""SQLite connection and schema. Pattern follows the course's board.py: WAL + busy_timeout."""

import json
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from lodestar.db.status import STATUSES
from lodestar.paths import db_path
from lodestar.schemas.common import Seniority, WorkMode
from lodestar.schemas.fit import LlmCall, Recommendation
from lodestar.schemas.job import ClassificationLayer, Source

log = logging.getLogger(__name__)

SCHEMA_VERSION = 7

RUN_KINDS = ("workflow", "eval", "batch")
DISCOVERY_VERDICTS = ("queued", "skipped", "invalid", "known")
LLM_CALL_STATUSES = get_args(LlmCall.model_fields["status"].annotation)


def _in(column: str, values, nullable: bool = False) -> str:
    """CHECK clause built from a Literal's values, so Python and SQL share one list."""
    allowed = ", ".join(f"'{v}'" for v in values)
    clause = f"{column} IN ({allowed})"
    return f"CHECK ({column} IS NULL OR {clause})" if nullable else f"CHECK ({clause})"


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,
    source        TEXT NOT NULL {_in("source", get_args(Source))},
    classified_by TEXT NOT NULL {_in("classified_by", get_args(ClassificationLayer))},
    title         TEXT NOT NULL,
    company       TEXT,
    location      TEXT,
    work_mode     TEXT {_in("work_mode", get_args(WorkMode), nullable=True)},
    seniority     TEXT {_in("seniority", get_args(Seniority), nullable=True)},
    salary_text   TEXT,
    date_posted   TEXT,
    valid_through TEXT,
    description   TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    status        TEXT NOT NULL {_in("status", STATUSES)},
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    dismissed_reason TEXT          -- set when a queued job is rejected without analysis
);

CREATE TABLE IF NOT EXISTS fit_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL REFERENCES jobs(id),
    overall_score  REAL NOT NULL,
    recommendation TEXT NOT NULL {_in("recommendation", get_args(Recommendation))},
    summary        TEXT NOT NULL,
    model          TEXT,
    created_at     TEXT NOT NULL,
    run_id         INTEGER REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_fit_results_job ON fit_results(job_id);

CREATE TABLE IF NOT EXISTS requirement_matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_result_id INTEGER NOT NULL REFERENCES fit_results(id),
    position      INTEGER NOT NULL,
    requirement   TEXT NOT NULL,
    priority      TEXT NOT NULL {_in("priority", ("must_have", "nice_to_have"))},
    match_level   TEXT NOT NULL {_in("match_level", ("direct", "related", "gap"))},
    evidence      TEXT NOT NULL,  -- JSON list
    explanation   TEXT NOT NULL,
    hard_constraint INTEGER NOT NULL DEFAULT 0 CHECK (hard_constraint IN (0, 1)),
    UNIQUE (fit_result_id, position)
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL {_in("kind", RUN_KINDS)},
    label           TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    analyses_ok     INTEGER NOT NULL DEFAULT 0,
    analyses_failed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES runs(id),
    job_id          TEXT,
    fit_result_id   INTEGER REFERENCES fit_results(id),
    model           TEXT NOT NULL,
    attempt         INTEGER NOT NULL,
    status          TEXT NOT NULL {_in("status", LLM_CALL_STATUSES)},
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    thinking_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER,
    error           TEXT,
    list_cost_usd   REAL,              -- tokens x list price; NULL when the price is unknown
    cost_usd        REAL,              -- what you pay: 0 on a free tier; NULL when unknown
    price_source    TEXT,
    started_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);

-- Every posting `lodestar discover` has seen, with the pre-filter's verdict. Doubles as
-- the "seen" registry: a posting already here isn't filtered again unless --recheck.
CREATE TABLE IF NOT EXISTS discoveries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL UNIQUE,
    url           TEXT NOT NULL,
    company       TEXT,
    ats           TEXT NOT NULL,
    board         TEXT NOT NULL,
    title         TEXT NOT NULL,
    location      TEXT,
    work_mode     TEXT,
    verdict       TEXT NOT NULL {_in("verdict", DISCOVERY_VERDICTS)},
    reason        TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_discoveries_board ON discoveries(ats, board);

CREATE TABLE IF NOT EXISTS resumes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL REFERENCES jobs(id),
    fit_result_id  INTEGER REFERENCES fit_results(id),
    content        TEXT NOT NULL,      -- TailoredResume JSON, as checked
    model          TEXT,
    prompt_version TEXT,
    created_at     TEXT NOT NULL,
    run_id         INTEGER REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_resumes_job ON resumes(job_id);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fit_result_id INTEGER NOT NULL REFERENCES fit_results(id),
    decision      TEXT NOT NULL {_in("decision", ("approve", "reject"))},
    note          TEXT,
    created_at    TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    """LODESTAR_DB if set, else data/lodestar.sqlite under the project root."""
    return db_path()


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database with WAL, busy_timeout and foreign keys on."""
    path = Path(path) if path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_v3(conn: sqlite3.Connection) -> None:
    """strong_fit / possible_fit / weak_fit -> top_pick / possible / blocked.

    SQLite can't change a CHECK constraint in place, so fit_results is rebuilt. Scores
    and recommendations are recomputed from each result's requirement rows with the
    current scoring rules (hard constraints out of the score; blocked only for an unmet
    hard constraint), rather than renamed: an old "weak_fit" could mean either a low
    score or a blocker.
    """
    from lodestar.schemas.fit import FitAnalysis, RequirementMatch
    from lodestar.scoring import score_analysis, unmet_hard_constraints

    conn.execute(f"""CREATE TABLE fit_results_v3 (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id         TEXT NOT NULL REFERENCES jobs(id),
        overall_score  REAL NOT NULL,
        recommendation TEXT NOT NULL {_in("recommendation", get_args(Recommendation))},
        summary        TEXT NOT NULL,
        model          TEXT,
        created_at     TEXT NOT NULL
    )""")
    for fit in conn.execute("SELECT * FROM fit_results ORDER BY id").fetchall():
        rows = conn.execute(
            "SELECT * FROM requirement_matches WHERE fit_result_id = ? ORDER BY position", (fit["id"],)
        ).fetchall()
        analysis = FitAnalysis(job_id=fit["job_id"], summary=fit["summary"], requirements=[
            RequirementMatch(requirement=r["requirement"], priority=r["priority"], evidence=json.loads(r["evidence"]),
                             match_level=r["match_level"], explanation=r["explanation"],
                             hard_constraint=bool(r["hard_constraint"]))
            for r in rows
        ])
        try:
            result = score_analysis(analysis)
            score, recommendation = result.overall_score, result.recommendation
        except ValueError:  # no skill requirements to score: keep the old score
            score = fit["overall_score"]
            recommendation = "blocked" if unmet_hard_constraints(analysis) else "possible"
        conn.execute(
            "INSERT INTO fit_results_v3 (id, job_id, overall_score, recommendation, summary, model, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fit["id"], fit["job_id"], score, recommendation, fit["summary"], fit["model"], fit["created_at"]),
        )
    conn.execute("DROP TABLE fit_results")
    conn.execute("ALTER TABLE fit_results_v3 RENAME TO fit_results")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fit_results_job ON fit_results(job_id)")


def _migrate_v4(conn: sqlite3.Connection) -> None:
    """Usage tracking: runs and llm_calls tables (created by SCHEMA), fit_results.run_id."""
    conn.execute("ALTER TABLE fit_results ADD COLUMN run_id INTEGER REFERENCES runs(id)")


# Steps that bring an existing database from version N-1 to N. A fresh database gets
# the full SCHEMA above and skips these.
MIGRATIONS: dict[int, str | Callable[[sqlite3.Connection], None] | None] = {
    2: "ALTER TABLE requirement_matches ADD COLUMN hard_constraint INTEGER NOT NULL DEFAULT 0 "
       "CHECK (hard_constraint IN (0, 1))",
    3: _migrate_v3,
    4: _migrate_v4,
    5: None,  # v5 only adds the discoveries table, which SCHEMA creates
    6: "ALTER TABLE jobs ADD COLUMN dismissed_reason TEXT",
    7: None,  # v7 only adds the resumes table, which SCHEMA creates
}


def _has_tables(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'").fetchone() is not None


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if missing, migrate an older database, and record the schema version."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema v{version} is newer than this code (v{SCHEMA_VERSION})")
    if version and version < SCHEMA_VERSION and _has_tables(conn):
        # Table rebuilds need foreign keys off (dropping a referenced table would fail);
        # the pragma only takes effect outside a transaction, and is checked afterwards.
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            with conn:
                for target in range(version + 1, SCHEMA_VERSION + 1):
                    step = MIGRATIONS[target]
                    if callable(step):
                        step(conn)
                    elif step:
                        conn.execute(step)
                    conn.execute(f"PRAGMA user_version = {target}")
                    log.info("migrated database schema v%d -> v%d", target - 1, target)
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise RuntimeError(f"foreign key problems after migration: {[tuple(p) for p in problems]}")
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    if version < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if not version:
            log.info("initialized database schema v%d", SCHEMA_VERSION)
    conn.commit()
