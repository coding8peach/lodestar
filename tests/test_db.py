import sqlite3
from datetime import date, datetime, timezone

import pytest

from lodestar.db import (
    SCHEMA_VERSION,
    InvalidTransition,
    connect,
    get_job,
    get_latest_fit_result,
    get_status,
    init_db,
    save_decision,
    save_fit_result,
    save_job,
    set_status,
)
from lodestar.schemas import FitAnalysis, FitResult, JobPosting, RequirementMatch, job_id_from_url

URL = "https://boards.greenhouse.io/acme/jobs/1"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    init_db(c)
    yield c
    c.close()


def make_job(url: str = URL) -> JobPosting:
    return JobPosting(
        id=job_id_from_url(url), url=url, source="pasted_url", classified_by="url_rule",
        title="Senior Backend Engineer", company="Acme", location="Remote - US", work_mode="remote",
        salary_text="USD 150,000–190,000", date_posted=date(2026, 9, 1), valid_through=None,
        description="Build agent tools.\n\n- Python\n- PostgreSQL",
        fetched_at=datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc),
    )


def make_result(job_id: str) -> FitResult:
    matches = [
        RequirementMatch(requirement="5+ years Python", priority="must_have", evidence=["proj-lodestar", "exp-a"],
                         match_level="related", explanation="Recent project Python; long Java career."),
        RequirementMatch(requirement="PostgreSQL", priority="must_have", evidence=["proj-sat-mcp"],
                         match_level="direct", explanation="Supabase Postgres."),
        RequirementMatch(requirement="Kubernetes", priority="nice_to_have", evidence=[],
                         match_level="gap", explanation="No evidence."),
    ]
    return FitResult(job_id=job_id, analysis=FitAnalysis(job_id=job_id, requirements=matches, summary="Solid."),
                     overall_score=0.72, recommendation="possible")


# --- connection -------------------------------------------------------------

def test_pragmas(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_init_db_is_idempotent(conn):
    init_db(conn)
    init_db(conn)


def test_newer_schema_refused(conn):
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer"):
        init_db(conn)


# --- jobs -------------------------------------------------------------------

def test_job_round_trip(conn):
    job = make_job()
    assert save_job(conn, job) is True
    assert get_job(conn, job.id) == job
    assert get_status(conn, job.id) == "queued"


def test_second_save_ignored(conn):
    job = make_job()
    save_job(conn, job)
    changed = job.model_copy(update={"title": "Changed"})
    assert save_job(conn, changed) is False
    assert get_job(conn, job.id).title == "Senior Backend Engineer"


def test_missing_job(conn):
    assert get_job(conn, "job_0000000000000000") is None
    assert get_status(conn, "job_0000000000000000") is None


def test_check_constraint_rejects_bad_status(conn):
    save_job(conn, make_job())
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("UPDATE jobs SET status = 'bogus'")


# --- status transitions -----------------------------------------------------

def test_invalid_transition(conn):
    job = make_job()
    save_job(conn, job)
    with pytest.raises(InvalidTransition, match="can't move from 'queued' to 'approved'"):
        set_status(conn, job.id, "approved")
    assert get_status(conn, job.id) == "queued"


def test_transition_on_missing_job(conn):
    with pytest.raises(InvalidTransition, match="not found"):
        set_status(conn, "job_0000000000000000", "analyzed")


# --- fit results ------------------------------------------------------------

def test_fit_result_round_trip(conn):
    job = make_job()
    save_job(conn, job)
    result = make_result(job.id)
    fit_id = save_fit_result(conn, result, model="gemini-test")
    assert get_status(conn, job.id) == "analyzed"
    got_id, got = get_latest_fit_result(conn, job.id)
    assert got_id == fit_id
    assert got == result  # order and evidence lists preserved
    assert conn.execute("SELECT model FROM fit_results").fetchone()[0] == "gemini-test"


def test_fit_result_needs_queued_job_and_rolls_back(conn):
    job = make_job()
    save_job(conn, job)
    save_fit_result(conn, make_result(job.id))
    with pytest.raises(InvalidTransition):
        save_fit_result(conn, make_result(job.id))  # already analyzed
    assert conn.execute("SELECT COUNT(*) FROM fit_results").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM requirement_matches").fetchone()[0] == 3


def test_fit_result_for_unknown_job_fails(conn):
    with pytest.raises(InvalidTransition, match="not found"):
        save_fit_result(conn, make_result("job_0000000000000000"))


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("INSERT INTO decisions (fit_result_id, decision, created_at) VALUES (999, 'approve', 'x')")


# --- decisions --------------------------------------------------------------

def test_decision_flow(conn):
    job = make_job()
    save_job(conn, job)
    fit_id = save_fit_result(conn, make_result(job.id))
    with pytest.raises(InvalidTransition):
        save_decision(conn, fit_id, "approve")  # not awaiting review yet
    set_status(conn, job.id, "awaiting_review")
    save_decision(conn, fit_id, "reject", note="Mostly Go")
    assert get_status(conn, job.id) == "rejected"
    row = conn.execute("SELECT decision, note FROM decisions WHERE fit_result_id = ?", (fit_id,)).fetchone()
    assert (row["decision"], row["note"]) == ("reject", "Mostly Go")


def test_decision_on_unknown_fit_result(conn):
    with pytest.raises(ValueError, match="not found"):
        save_decision(conn, 999, "approve")


# --- schema v2: hard_constraint -------------------------------------------------------

def test_hard_constraint_round_trip(conn):
    job = make_job()
    save_job(conn, job)
    result = make_result(job.id)
    blocked = result.analysis.requirements[0].model_copy(update={"hard_constraint": True})
    result = result.model_copy(update={"analysis": result.analysis.model_copy(
        update={"requirements": [blocked, *result.analysis.requirements[1:]]})})
    save_fit_result(conn, result)
    _, got = get_latest_fit_result(conn, job.id)
    assert [m.hard_constraint for m in got.analysis.requirements] == [True, False, False]


def test_migrates_v1_database(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY);
        CREATE TABLE fit_results (id INTEGER PRIMARY KEY, job_id TEXT, overall_score REAL,
            recommendation TEXT, summary TEXT, model TEXT, created_at TEXT);
        INSERT INTO jobs VALUES ('job_1');
        INSERT INTO fit_results VALUES (1, 'job_1', 1.0, 'strong_fit', 's', 'm', 't');
        CREATE TABLE requirement_matches (id INTEGER PRIMARY KEY, fit_result_id INTEGER, position INTEGER,
            requirement TEXT, priority TEXT, match_level TEXT, evidence TEXT, explanation TEXT);
        INSERT INTO requirement_matches VALUES (1, 1, 0, 'Python', 'must_have', 'direct', '[]', 'x');
        PRAGMA user_version = 1;
    """)
    old.close()
    conn = connect(path)
    init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT hard_constraint FROM requirement_matches").fetchone()[0] == 0
    conn.close()


def test_migrates_v2_recommendations(tmp_path):
    """v2 -> v3: recommendations are recomputed from requirement rows, not renamed."""
    path = tmp_path / "v2.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY);
        CREATE TABLE fit_results (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
            overall_score REAL NOT NULL,
            recommendation TEXT NOT NULL CHECK (recommendation IN ('strong_fit', 'possible_fit', 'weak_fit')),
            summary TEXT NOT NULL, model TEXT, created_at TEXT NOT NULL);
        CREATE TABLE requirement_matches (id INTEGER PRIMARY KEY, fit_result_id INTEGER REFERENCES fit_results(id),
            position INTEGER, requirement TEXT, priority TEXT, match_level TEXT, evidence TEXT, explanation TEXT,
            hard_constraint INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE decisions (id INTEGER PRIMARY KEY, fit_result_id INTEGER REFERENCES fit_results(id),
            decision TEXT, note TEXT, created_at TEXT);
        INSERT INTO jobs VALUES ('job_a'), ('job_b'), ('job_c');
        -- weak because of a low score -> possible
        INSERT INTO fit_results VALUES (1, 'job_a', 0.15, 'weak_fit', 's', 'm', 't');
        INSERT INTO requirement_matches VALUES (1, 1, 0, 'Go', 'must_have', 'gap', '[]', '', 0),
                                               (2, 1, 1, 'SQL', 'must_have', 'related', '[]', '', 0);
        -- weak because of a blocker -> blocked (skill score recomputed without it)
        INSERT INTO fit_results VALUES (2, 'job_b', 0.8, 'weak_fit', 's', 'm', 't');
        INSERT INTO requirement_matches VALUES (3, 2, 0, 'Java', 'must_have', 'direct', '[]', '', 0),
                                               (4, 2, 1, 'Canada', 'must_have', 'gap', '[]', '', 1);
        -- strong -> top_pick
        INSERT INTO fit_results VALUES (3, 'job_c', 0.9, 'strong_fit', 's', 'm', 't');
        INSERT INTO requirement_matches VALUES (5, 3, 0, 'Java', 'must_have', 'direct', '[]', '', 0);
        INSERT INTO decisions VALUES (1, 2, 'reject', 'Canada', 't');
        PRAGMA user_version = 2;
    """)
    old.close()
    conn = connect(path)
    init_db(conn)
    rows = conn.execute("SELECT id, overall_score, recommendation FROM fit_results ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(1, 0.3, "possible"), (2, 1.0, "blocked"), (3, 1.0, "top_pick")]
    assert conn.execute("SELECT fit_result_id FROM decisions").fetchone()[0] == 2  # links intact
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("UPDATE fit_results SET recommendation = 'weak_fit' WHERE id = 1")
    conn.close()


# --- schema v4: usage tracking ----------------------------------------------------------

def _call(model, status="ok", tokens_in=1000, tokens_out=100):
    from lodestar.schemas import LlmCall
    return LlmCall(model=model, attempt=1, status=status, input_tokens=tokens_in, output_tokens=tokens_out,
                   started_at=datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc))


def test_usage_by_run_day_and_model(conn, monkeypatch):
    from lodestar import pricing
    from lodestar.db import finish_run, save_llm_calls, start_run, usage_by

    monkeypatch.setattr(pricing, "_overrides", lambda: {
        "paid": {"input_per_mtok": 1.0, "output_per_mtok": 10.0},
        "free": {"input_per_mtok": 1.0, "output_per_mtok": 10.0, "free_tier": True}})
    run_id = start_run(conn, "eval", "prompt-v2c-r1")
    save_llm_calls(conn, [_call("free", "rate_limited", 0, 0), _call("paid"), _call("mystery")], run_id, "job_1")
    finish_run(conn, run_id, analyses_ok=1, analyses_failed=0)

    (row,) = usage_by(conn, "run")
    assert (row["kind"], row["label"], row["analyses_ok"]) == ("eval", "prompt-v2c-r1", 1)
    assert (row["calls"], row["failed_calls"], row["input_tokens"]) == (3, 1, 2000)
    assert row["list_cost_usd"] == row["cost_usd"] == 0.002        # paid: 1000 x 1 + 100 x 10 per 1M
    assert row["unpriced_calls"] == 1                                # mystery: unknown, not $0
    models = {r["model"]: r for r in usage_by(conn, "model")}
    assert models["free"]["cost_usd"] == 0.0 and models["mystery"]["cost_usd"] is None
    assert usage_by(conn, "day")[0]["day"] == "2026-09-28"


def test_fit_result_links_to_run(conn):
    from lodestar.db import start_run
    job = make_job()
    save_job(conn, job)
    run_id = start_run(conn, "workflow", job.url)
    fit_id = save_fit_result(conn, make_result(job.id), model="m", run_id=run_id)
    assert conn.execute("SELECT run_id FROM fit_results WHERE id = ?", (fit_id,)).fetchone()[0] == run_id


def test_migrates_v3_to_v4(tmp_path):
    path = tmp_path / "v3.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY);
        CREATE TABLE fit_results (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
            overall_score REAL NOT NULL, recommendation TEXT NOT NULL, summary TEXT NOT NULL, model TEXT,
            created_at TEXT NOT NULL);
        INSERT INTO jobs VALUES ('job_a');
        INSERT INTO fit_results VALUES (1, 'job_a', 0.8, 'top_pick', 's', 'm', 't');
        PRAGMA user_version = 3;
    """)
    old.close()
    conn = connect(path)
    init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT run_id FROM fit_results").fetchone()[0] is None
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0
    conn.close()


def test_migrates_v5_to_v6(tmp_path):
    path = tmp_path / "v5.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT);
        INSERT INTO jobs VALUES ('job_a', 'queued');
        PRAGMA user_version = 5;
    """)
    old.close()
    conn = connect(path)
    init_db(conn)
    assert conn.execute("SELECT dismissed_reason FROM jobs").fetchone()[0] is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 6
    conn.close()
