"""Build the demo database from the fictional candidate and postings, using the real agents.

    uv run python demo/build_demo.py            # needs LODESTAR_FIT_MODELS and API keys in .env
    uv run python demo/build_demo.py --finish   # keep the database; only redo what failed

Writes demo/demo.sqlite (committed; the deployed demo serves a copy of it). Steps:
  1. every posting goes through the real pre-filter: passing ones are queued, the rest logged
     as skipped (so the funnel shows real filtering)
  2. the real fit agent analyzes the top ANALYZE jobs in ranking order (free models unless
     LODESTAR_PAID_USD_PER_DAY allows paid fallback)
  3. scripted decisions: blocked jobs rejected, top picks approved, one low fit rejected,
     the rest left waiting for review
  4. the real resume agent tailors resumes for the first TAILOR approved jobs
The agents run in this process (no `lodestar-agent` needed); the MCP server they use is
started with this script's LODESTAR_DB / LODESTAR_PROFILE, so it reads the demo data.
"""

import logging
import os
import sys
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DB = DEMO_DIR / "demo.sqlite"
ANALYZE = 9
TAILOR = 2
PAUSE_BETWEEN_JOBS = 5.0  # seconds between resumes: gives a busy free-tier model a moment
APPROVE_NOTES = ["Strong backend match; apply this week", "Agent work I'm doing already; good stretch role",
                 "Good fit; check team size"]


def configure_environment(db: Path = DB) -> None:
    os.environ["LODESTAR_DB"] = str(db)
    os.environ["LODESTAR_PROFILE"] = str(DEMO_DIR / "profile.yaml")
    os.environ.pop("LODESTAR_DEMO", None)  # building writes; the demo itself is read-only


def load_postings(db_conn) -> dict:
    """Pre-filter every posting; save passing ones as queued. Returns counts."""
    import yaml

    from lodestar.db import save_job
    from lodestar.db.status import now_utc
    from lodestar.ingest.prefilter import load_rules, prefilter
    from lodestar.schemas import JobPosting, job_id_from_url

    rules = load_rules()
    counts = {"queued": 0, "skipped": 0}
    postings = yaml.safe_load((DEMO_DIR / "postings.yaml").read_text(encoding="utf-8"))
    for i, p in enumerate(postings, 1):
        slug = p["company"].lower().replace(" ", "-")
        url = f"https://careers.example.com/{slug}/{i}"
        job = JobPosting(id=job_id_from_url(url), url=url, source="pasted_url", classified_by="json_ld",
                         title=p["title"], company=p["company"], location=p.get("location"),
                         work_mode=p.get("work_mode"), salary_text=p.get("salary_text"),
                         date_posted=date.today() - timedelta(days=i), description=p["description"].strip(),
                         fetched_at=datetime.now(timezone.utc))
        passes, reason = prefilter(job, rules)
        verdict = "queued" if passes else "skipped"
        counts[verdict] += 1
        if passes:
            save_job(db_conn, job)
        with db_conn:
            db_conn.execute(
                "INSERT INTO discoveries (job_id, url, company, ats, board, title, location, work_mode, verdict,"
                " reason, first_seen_at, last_seen_at) VALUES (?, ?, ?, 'demo', ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, url, job.company, slug, job.title, job.location, job.work_mode, verdict, reason,
                 now_utc(), now_utc()))
    return counts


def direct_agents(model_names: list[str]):
    """The fit and resume agents called in-process, shaped like the A2A clients."""
    from lodestar.fit_agent.runner import analyze_job
    from lodestar.fit_agent.service import allowed_models
    from lodestar.paths import profile_path
    from lodestar.resume_agent.runner import tailor_job
    from lodestar.schemas import FitAgentReply, load_profile
    from lodestar.schemas.resume import ResumeAgentReply

    async def analyze(job_id, max_paid_usd=None):
        run = await analyze_job(job_id, allowed_models(model_names, max_paid_usd))
        return FitAgentReply(analysis=run.analysis, model=run.model, skipped=run.skipped, tokens=run.tokens,
                             calls=run.calls)

    async def tailor(job_id, max_paid_usd=None):
        run = await tailor_job(job_id, allowed_models(model_names, max_paid_usd), load_profile(profile_path()))
        return ResumeAgentReply(resume=run.result, model=run.model, skipped=run.skipped, tokens=run.tokens,
                                calls=run.calls)

    return analyze, tailor


def decide() -> list[str]:
    """Scripted decisions. Returns approved job ids, best first."""
    from lodestar.app import service

    approved = []
    for item in service.review_queue():
        if item.recommendation == "blocked":
            service.record_decision(item.job_id, "reject", "blocked: " + "; ".join(item.blocked_by))
        elif item.recommendation == "top_pick" and len(approved) < len(APPROVE_NOTES):
            service.record_decision(item.job_id, "approve", APPROVE_NOTES[len(approved)])
            approved.append(item.job_id)
    waiting = [i for i in service.review_queue() if i.recommendation == "possible"]
    if waiting:  # one low fit rejected, to show a rejection with a note; the rest stay waiting
        low = min(waiting, key=lambda i: i.score)
        service.record_decision(low.job_id, "reject", "Mostly frontend and a stack I'd need to learn")
    return approved


def _tailor_missing(tailor_fn) -> int:
    """Tailor approved jobs without a resume until TAILOR have one. Returns how many have one."""
    import time

    from lodestar.app import service

    approved = [i for i in service.decisions("approve") if not i.dismissed]
    have = sum(1 for i in approved if i.has_resume)
    for item in approved:
        if have >= TAILOR:
            break
        if item.has_resume:
            continue
        try:
            service.tailor(item.job_id, tailor_fn=tailor_fn)
            have += 1
        except ValueError as e:
            logging.warning("tailoring %s failed: %s", item.job_id, e)
        time.sleep(PAUSE_BETWEEN_JOBS)
    return have


def _single_file() -> None:
    from lodestar.db import connect

    with closing(connect()) as conn:  # one self-contained file for the repo
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")


def finish(analyze_fn, tailor_fn, db: Path = DB) -> dict:
    """Keep the existing demo database; analyze what's still queued (up to ANALYZE in total) and
    tailor the missing resumes. For when a busy model made part of a build fail."""
    configure_environment(db)
    if not db.exists():
        raise FileNotFoundError(f"{db} doesn't exist yet; run without --finish first")
    from lodestar.app import service
    from lodestar.db import connect, init_db

    with closing(connect()) as conn:
        init_db(conn)
        analyzed = conn.execute("SELECT COUNT(DISTINCT job_id) FROM fit_results").fetchone()[0]
    batch = None
    if analyzed < ANALYZE:
        batch = service.analyze_next(ANALYZE - analyzed, analyze_fn=analyze_fn)
    if not service.decisions("approve"):
        decide()
    tailored = _tailor_missing(tailor_fn)
    _single_file()
    return {"analyzed": analyzed + (batch.ok if batch else 0), "failed": batch.failed if batch else 0,
            "approved": len(service.decisions("approve")), "tailored": tailored}


def build(analyze_fn, tailor_fn, db: Path = DB) -> dict:
    configure_environment(db)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)

    from lodestar.app import service
    from lodestar.db import connect, init_db

    with closing(connect()) as conn:
        init_db(conn)
        counts = load_postings(conn)
    batch = service.analyze_next(ANALYZE, analyze_fn=analyze_fn,
                                 on_progress=lambda j: logging.info("analyzed %s: %s", j.title, j.recommendation
                                                                    or j.error))
    approved = decide()
    tailored = _tailor_missing(tailor_fn)
    _single_file()
    return {**counts, "analyzed": batch.ok, "failed": batch.failed, "approved": len(approved),
            "tailored": tailored}


def main() -> int:
    from dotenv import load_dotenv

    from lodestar.paths import PROJECT_ROOT
    from lodestar.quiet import silence

    load_dotenv(PROJECT_ROOT / ".env")
    silence()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    silence()
    from lodestar.fit_agent.models import model_names_from_env

    analyze_fn, tailor_fn = direct_agents(model_names_from_env())
    result = (finish if "--finish" in sys.argv else build)(analyze_fn, tailor_fn)
    logging.info("demo database written to %s: %s", DB, result)
    if result["failed"] or result["tailored"] < TAILOR:
        logging.warning("some steps failed (often a busy model); later, run with --finish to redo only those")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
