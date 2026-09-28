"""Run the fit agent on golden jobs and show its judgment next to yours.

    uv run python eval/run_fit.py job_02
    uv run python eval/run_fit.py --all --label profile-v1
    uv run python eval/run_fit.py --all --label prompt-v2 --repeat 3   (→ prompt-v2-r1, -r2, -r3)
    uv run python eval/run_fit.py job_02 --models gemini-3.1-flash-lite
    uv run python eval/run_fit.py job_02 --via-a2a      (needs `uv run lodestar-agent` running)

For each job: the agent reads the job and profile through MCP, Python scores the
analysis, and the result is printed beside your expected.yaml decision and reason.
Models are tried in order, one whole run each (see fit_agent/runner.py).
Results are saved to eval/runs/<label>/<job>.json (label defaults to a timestamp).
No fit results or statuses are written to the database; a golden job missing from
the database is inserted first so the MCP server can serve it.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from lodestar.quiet import silence

silence()

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from lodestar.db import connect, finish_run, init_db, save_job, save_llm_calls, start_run  # noqa: E402
from lodestar.fit_agent.models import model_names_from_env, validate_model_names  # noqa: E402
from lodestar.fit_agent.parse import unknown_evidence  # noqa: E402
from lodestar.fit_agent.agent import PROMPT_VERSION  # noqa: E402
from lodestar.fit_agent.client import analyze_via_a2a  # noqa: E402
from lodestar.fit_agent.runner import FitRun, analyze_job  # noqa: E402
from lodestar.paths import PROJECT_ROOT, profile_path  # noqa: E402
from lodestar.report import fit_table, report_logger  # noqa: E402
from lodestar.schemas import JobPosting, load_profile  # noqa: E402
from lodestar.scoring import score_analysis  # noqa: E402

log = logging.getLogger("run_fit")

EVAL_DIR = Path(__file__).resolve().parent


def load_golden(name: str) -> JobPosting:
    return JobPosting.model_validate_json((EVAL_DIR / "jobs" / f"{name}.json").read_text(encoding="utf-8"))


def ensure_in_db(job: JobPosting) -> None:
    with closing(connect()) as conn:
        init_db(conn)
        if save_job(conn, job):
            log.info("inserted %s into the database so the MCP server can serve it", job.id)


async def analyze(job_id: str, model_names: list[str], via_a2a: bool) -> FitRun:
    if not via_a2a:
        return await analyze_job(job_id, model_names)
    reply = await analyze_via_a2a(job_id)  # the service picks models from its own LODESTAR_FIT_MODELS
    return FitRun(analysis=reply.analysis, model=reply.model, tokens=reply.tokens or 0, skipped=reply.skipped,
                  calls=reply.calls)


async def main_async(names: list[str], label: str, model_names: list[str], via_a2a: bool = False) -> int:
    expected_all = yaml.safe_load((EVAL_DIR / "expected.yaml").read_text(encoding="utf-8")) or {}
    profile = load_profile(profile_path())
    out_dir = EVAL_DIR / "runs" / label
    with closing(connect()) as conn:
        init_db(conn)
        run_id = start_run(conn, "eval", label)
    failures, successes = 0, 0
    for name in names:
        job = load_golden(name)
        ensure_in_db(job)
        try:
            run = await analyze(job.id, model_names, via_a2a)
        except Exception as e:
            calls = getattr(e, "calls", None)  # AllModelsFailed carries the failed attempts
            if calls:
                with closing(connect()) as conn:
                    save_llm_calls(conn, calls, run_id, job.id)
            first_line = (str(e).splitlines() or [""])[0]
            log.error("%s failed: %s: %s", name, type(e).__name__, first_line[:300])
            failures += 1
            continue
        successes += 1
        with closing(connect()) as conn:
            save_llm_calls(conn, run.calls, run_id, job.id)
        result = score_analysis(run.analysis)
        bad_ids = unknown_evidence(run.analysis, profile)
        for line in fit_table(job, result, run.model, heading=name, tokens=run.tokens, skipped=run.skipped,
                              expected=expected_all.get(name) or {}, unknown_evidence=bad_ids):
            report_logger().info(line)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps({
            "job": name,
            "job_id": job.id,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "model": run.model,
            # With --via-a2a this is the local code's version; the service must be restarted to match.
            "prompt_version": PROMPT_VERSION,
            "skipped": run.skipped,
            "tokens": run.tokens,
            "result": result.model_dump(),
            "unknown_evidence": bad_ids,
            "expected": expected_all.get(name),
        }, indent=2) + "\n", encoding="utf-8")
    with closing(connect()) as conn:
        finish_run(conn, run_id, successes, failures)
    if out_dir.exists():
        log.info("saved results to %s", out_dir)
    log.info("usage recorded as run #%d (see: uv run lodestar usage)", run_id)
    return 1 if failures else 0


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    silence()  # again, after basicConfig

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="*", help="golden job names, e.g. job_02")
    parser.add_argument("--all", action="store_true", help="run every job in eval/jobs/")
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d-%H%M"), help="run folder name")
    parser.add_argument("--models", help="comma-separated model list (overrides LODESTAR_FIT_MODELS)")
    parser.add_argument("--via-a2a", action="store_true", help="ask the running fit agent service instead")
    parser.add_argument("--repeat", type=int, default=1, help="run the set N times, saved as <label>-r1 .. -rN")
    args = parser.parse_args()
    if args.via_a2a and args.models:
        parser.error("--models can't be combined with --via-a2a; the service uses its own LODESTAR_FIT_MODELS")

    names = sorted(p.stem for p in (EVAL_DIR / "jobs").glob("job_*.json")) if args.all else args.jobs
    if not names:
        parser.error("name at least one job, or use --all")
    if not re.fullmatch(r"[\w.-]+", args.label):
        parser.error("label may contain only letters, digits, '.', '_' and '-'")
    try:
        if args.models:
            model_names = validate_model_names([m.strip() for m in args.models.split(",") if m.strip()])
        else:
            model_names = model_names_from_env()
    except ValueError as e:
        parser.error(str(e))
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be between 1 and 10")
    labels = [args.label] if args.repeat == 1 else [f"{args.label}-r{i}" for i in range(1, args.repeat + 1)]
    status = 0
    for label in labels:
        if len(labels) > 1:
            log.info("run %s", label)
        status |= asyncio.run(main_async(names, label, model_names, args.via_a2a))
    return status


if __name__ == "__main__":
    sys.exit(main())
