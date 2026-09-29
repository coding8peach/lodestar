"""Lodestar from the terminal.

    uv run lodestar "<job posting URL>"        ingest a posting, analyze the fit, review it
    uv run lodestar usage [--by run|day|model] model calls, tokens and estimated cost
    uv run lodestar discover [--dry-run]       list watched job boards, pre-filter, queue (no LLM)
    uv run lodestar analyze [--budget N]       rank the queue, analyze the top N (--dry-run: rank only)
    uv run lodestar review [--limit N]         decide on analyzed jobs, best first
    uv run lodestar review --dismiss JOB_ID    reject a queued job without analyzing it
    uv run lodestar ui                         open the local web UI (Review, Queue, Decisions, Usage)
    uv run lodestar tailor JOB_ID              draft a resume tailored to an approved job

The fit agent runs as its own service; start it first in another terminal:
    uv run lodestar-agent

Diagnostic logs go to stderr and to data/logs/lodestar.log; the fit table goes to stdout.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from contextlib import closing

from lodestar.quiet import silence

silence()

from dotenv import load_dotenv  # noqa: E402
from langgraph.types import Command  # noqa: E402

from lodestar.db import connect, finish_run, get_job, get_status, init_db, start_run, usage_by  # noqa: E402
from lodestar.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402
from lodestar.report import fit_table, report_logger  # noqa: E402
from lodestar.schemas.fit import FitResult  # noqa: E402
from lodestar.workflow.graph import build_graph  # noqa: E402

CHOICES = {"a": "approve", "approve": "approve", "r": "reject", "reject": "reject", "s": "skip", "skip": "skip",
           "q": "quit", "quit": "quit"}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "lodestar.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    silence()


def ask_decision(allow_quit: bool = False) -> dict:
    """Prompt until we get approve / reject / skip (or quit). End of input (Ctrl-D) counts as skip."""
    prompt = "\nDecision [a]pprove / [r]eject / [s]kip" + (" / [q]uit" if allow_quit else "") + ": "
    try:
        while True:
            raw = input(prompt).strip().lower()
            if raw in CHOICES and (allow_quit or CHOICES[raw] != "quit"):
                decision = CHOICES[raw]
                break
            report_logger().info("Please type a, r, s" + (" or q." if allow_quit else " or s."))
        note = input("Note (optional): ").strip() if decision in ("approve", "reject") else ""
    except EOFError:
        return {"decision": "skip", "note": None}
    return {"decision": decision, "note": note or None}


async def run(url: str) -> int:
    with closing(connect()) as conn:
        init_db(conn)
        run_id = start_run(conn, "workflow", url)
    try:
        return await _run(url, run_id)
    finally:
        # One job per workflow run: it counts as analyzed if a fit result exists.
        with closing(connect()) as conn:
            analyzed = conn.execute("SELECT COUNT(*) FROM fit_results WHERE run_id = ?", (run_id,)).fetchone()[0]
            finish_run(conn, run_id, analyses_ok=analyzed, analyses_failed=0 if analyzed else 1)


async def _run(url: str, run_id: int) -> int:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = report_logger()

    state = await graph.ainvoke({"url": url, "run_id": run_id}, config)
    if state.get("error"):
        return 1  # the failing node already logged the reason

    with closing(connect()) as conn:
        init_db(conn)
        job = get_job(conn, state["job_id"])
    for line in fit_table(job, FitResult.model_validate(state["fit_result"]), state.get("model")):
        out.info(line)
    if state.get("reused_analysis"):
        out.info("(earlier analysis)")

    if state.get("__interrupt__"):
        answer = await asyncio.to_thread(ask_decision)
        state = await graph.ainvoke(Command(resume=answer), config)
        if state.get("error"):
            return 1

    with closing(connect()) as conn:
        status = get_status(conn, state["job_id"])
    if state.get("already_decided"):
        note = f" ({state['note']})" if state.get("note") else ""
        out.info(f"\nalready decided: {state.get('decision')}{note}")
    out.info(f"job {state['job_id']}: status {status}, fit result {state['fit_result_id']}")
    return 0


def _money(value) -> str:
    return "-" if value is None else f"${value:.4f}"


def usage(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lodestar usage", description="Model calls, tokens and estimated cost.")
    parser.add_argument("--by", choices=["run", "day", "model"], default="run")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    with closing(connect()) as conn:
        init_db(conn)
        rows = usage_by(conn, args.by, args.limit)
    out = report_logger()
    if not rows:
        out.info("no usage recorded yet")
        return 0
    key = {"run": "run", "day": "day", "model": "model"}[args.by]
    out.info(f"{key:<32}{'calls':>6}{'failed':>8}{'in tok':>10}{'out tok':>10}{'list cost':>12}{'you pay':>11}")
    out.info("-" * 89)
    totals = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0,
              "list_cost_usd": None, "cost_usd": None, "unpriced_calls": 0}
    for r in rows:
        if args.by == "run":
            name = f"#{r['run_id']} {r['kind']} {r['started_at'][:16]}"
        else:
            name = str(r[key])
        out.info(f"{name[:31]:<32}{r['calls']:>6}{r['failed_calls'] or 0:>8}{r['input_tokens']:>10}"
                 f"{r['output_tokens']:>10}{_money(r['list_cost_usd']):>12}{_money(r['cost_usd']):>11}")
        for k in totals:
            if k in ("list_cost_usd", "cost_usd"):
                if r[k] is not None:  # unknown stays unknown: "-" rather than $0
                    totals[k] = (totals[k] or 0.0) + r[k]
            else:
                totals[k] += r[k] or 0
    out.info("-" * 89)
    out.info(f"{'total (rows shown)':<32}{totals['calls']:>6}{totals['failed_calls']:>8}{totals['input_tokens']:>10}"
             f"{totals['output_tokens']:>10}{_money(totals['list_cost_usd']):>12}{_money(totals['cost_usd']):>11}")
    out.info("list cost = tokens x list price; you pay = 0 for models marked free_tier in config/prices.yaml")
    if totals["unpriced_calls"]:
        out.info(f"{totals['unpriced_calls']} call(s) have no known price and are not in the cost totals "
                 "(add them to config/prices.yaml)")
    return 0


def discover_cmd(argv: list[str]) -> int:
    from lodestar.app import service

    parser = argparse.ArgumentParser(prog="lodestar discover",
                                     description="List watched job boards, pre-filter, queue passing jobs. No LLM.")
    parser.add_argument("--dry-run", action="store_true", help="list and filter, save nothing")
    parser.add_argument("--recheck", action="store_true", help="re-filter postings skipped in earlier runs")
    args = parser.parse_args(argv)

    summary = service.discover(dry_run=args.dry_run, recheck=args.recheck)
    out = report_logger()
    out.info("")
    out.info(f"{'company':<16}{'listed':>7}{'seen':>6}{'new':>6}{'queued':>8}   top skip reasons")
    out.info("-" * 96)
    for b in summary.boards:
        if b.error:
            out.info(f"{b.company[:15]:<16}  error: {b.error}")
            continue
        top = ", ".join(f"{reason} {n}" for reason, n in sorted(b.skipped.items(), key=lambda kv: -kv[1])[:3])
        out.info(f"{b.company[:15]:<16}{b.listed:>7}{b.seen:>6}{b.listed - b.seen:>6}{b.queued:>8}   {top}")
    out.info("-" * 96)
    out.info(f"{'total':<16}{summary.listed:>7}{summary.seen:>6}{summary.listed - summary.seen:>6}"
             f"{len(summary.queued_jobs):>8}")
    if summary.queued_jobs:
        out.info("")
        out.info("dry run: would queue" if args.dry_run else "queued for analysis:")
        for company, title, location in sorted(summary.queued_jobs, key=lambda j: (j[0] or "", j[1])):
            out.info(f"  {company or '?':<14} {title[:52]:<53} {location or '-'}")
    if summary.skipped:
        out.info("")
        out.info("skipped: " + ", ".join(f"{reason} {n}" for reason, n in summary.skipped.items()))
    if args.dry_run:
        out.info("\n(dry run: nothing saved)")
    all_failed = all(b.error for b in summary.boards) and summary.boards
    return 1 if all_failed else 0


def analyze_cmd(argv: list[str]) -> int:
    import os

    from lodestar.app import service
    from lodestar.paths import profile_path
    from lodestar.ranking import load_queue, load_rank_config, pick, rank_queue
    from lodestar.schemas import load_profile

    parser = argparse.ArgumentParser(prog="lodestar analyze",
                                     description="Rank queued jobs (no LLM) and analyze the top N.")
    parser.add_argument("--budget", type=int, default=3, help="analyses this run (default 3)")
    parser.add_argument("--dry-run", action="store_true", help="show the ranking, analyze nothing")
    parser.add_argument("--show", type=int, default=15, help="ranked roles to list")
    args = parser.parse_args(argv)
    if args.budget < 0:
        parser.error("--budget can't be negative")
    try:
        float(os.environ.get("LODESTAR_PAID_USD_PER_DAY", "0") or 0)
    except ValueError:
        parser.error("LODESTAR_PAID_USD_PER_DAY must be a number of dollars, e.g. 0.10")

    config = load_rank_config()
    with closing(connect()) as conn:
        init_db(conn)
        queued, analyzed = load_queue(conn)
    ranked, already = rank_queue(queued, analyzed, load_profile(profile_path()), config)
    chosen = {r.job.id for r in pick(ranked, args.budget, config)}

    out = report_logger()
    out.info("")
    out.info(f"{len(queued)} queued: {len(ranked)} distinct roles, "
             f"{sum(len(r.similar) for r in ranked)} near-duplicates, {len(already)} similar to analyzed jobs")
    out.info("")
    out.info(f"    {'score':>5}  {'company':<13} {'title':<52} matched")
    for i, r in enumerate(ranked[:args.show], 1):
        mark = "->" if r.job.id in chosen else "  "
        similar = f" (+{len(r.similar)} similar)" if r.similar else ""
        out.info(f"{mark}{i:>2} {r.score:>5}  {(r.job.company or '?')[:12]:<13} {r.job.title[:51]:<52} "
                 f"{', '.join(r.matched[:6])}{similar}")
    if len(ranked) > args.show:
        out.info(f"   ... {len(ranked) - args.show} more")
    if args.dry_run or not chosen:
        out.info("\n(dry run: nothing analyzed)" if args.dry_run else "\nnothing to analyze")
        return 0

    out.info("")

    def progress(item) -> None:
        if item.error:
            out.info(f"  failed    {item.company or '?'} | {item.title}: {item.error}")
        else:
            out.info(f"  {item.recommendation:<9} {item.score:.2f}  {item.company or '?'} | {item.title}")

    batch = service.analyze_next(args.budget, on_progress=progress)
    if batch.stopped:
        out.info(f"stopping: {batch.stopped}")
    out.info(f"\nanalyzed {batch.ok}, failed {batch.failed}; paid-model limit today ${batch.paid_limit_usd:.2f} "
             "(LODESTAR_PAID_USD_PER_DAY); see: uv run lodestar usage")
    return 0 if batch.ok or not batch.failed else 1


def review_cmd(argv: list[str]) -> int:
    from lodestar.db import dismiss_job, get_latest_fit_result, review_queue
    from lodestar.scoring import unmet_hard_constraints

    parser = argparse.ArgumentParser(prog="lodestar review", description="Decide on analyzed jobs, best first.")
    parser.add_argument("--limit", type=int, default=None, help="review at most N jobs (blocked ones excluded)")
    parser.add_argument("--dismiss", metavar="JOB_ID", help="reject a queued job without analyzing it")
    parser.add_argument("--reason", default="dismissed without analysis", help="reason stored with --dismiss")
    args = parser.parse_args(argv)
    out = report_logger()

    if args.dismiss:
        from lodestar.db import InvalidTransition
        with closing(connect()) as conn:
            init_db(conn)
            try:
                dismiss_job(conn, args.dismiss, args.reason)
            except InvalidTransition as e:
                out.info(f"can't dismiss: {e} (only queued jobs can be dismissed)")
                return 1
        out.info(f"dismissed {args.dismiss}: {args.reason}")
        return 0

    with closing(connect()) as conn:
        init_db(conn)
        pending = review_queue(conn)
    to_review = [p for p in pending if p["recommendation"] != "blocked"]
    blocked = [p for p in pending if p["recommendation"] == "blocked"]
    if args.limit is not None:
        to_review = to_review[:args.limit]
    if not pending:
        out.info("nothing to review: no analyzed jobs are waiting")
        return 0
    out.info(f"{len(to_review)} to review" + (f", {len(blocked)} blocked" if blocked else ""))

    counts = {"approve": 0, "reject": 0, "skip": 0}

    async def go() -> None:
        graph = build_graph()
        for i, item in enumerate(to_review, 1):
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            state = await graph.ainvoke({"job_id": item["job_id"]}, config)
            if state.get("error"):
                continue
            with closing(connect()) as conn:
                job = get_job(conn, item["job_id"])
            out.info(f"\n[{i}/{len(to_review)}]")
            for line in fit_table(job, FitResult.model_validate(state["fit_result"]), state.get("model")):
                out.info(line)
            out.info(f"url: {job.url}")
            if not state.get("__interrupt__"):
                continue
            answer = await asyncio.to_thread(ask_decision, True)
            if answer["decision"] == "quit":
                out.info("stopped; the rest stay waiting for review")
                return
            state = await graph.ainvoke(Command(resume=answer), config)
            if not state.get("error"):
                counts[answer["decision"]] += 1

    asyncio.run(go())

    if blocked:
        out.info("")
        out.info(f"{len(blocked)} blocked (a hard constraint isn't met):")
        with closing(connect()) as conn:
            for item in blocked:
                latest = get_latest_fit_result(conn, item["job_id"])
                why = "; ".join(r.requirement for r in unmet_hard_constraints(latest[1].analysis)) if latest else "?"
                out.info(f"  {item['company'] or '?'} | {item['title']} -- {why}")
        try:
            bulk = input(f"Reject all {len(blocked)} blocked jobs? [y/N]: ").strip().lower() == "y"
        except EOFError:
            bulk = False
        if bulk:
            from lodestar.app.service import reject_blocked

            done = reject_blocked(item["job_id"] for item in blocked)
            counts["reject"] += done.done

    with closing(connect()) as conn:
        waiting = len(review_queue(conn))
        queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]
    out.info(f"\napproved {counts['approve']}, rejected {counts['reject']}, skipped {counts['skip']}; "
             f"{waiting} still waiting for review, {queued} queued for analysis")
    return 0


def tailor_cmd(argv: list[str]) -> int:
    from lodestar.app import service

    parser = argparse.ArgumentParser(prog="lodestar tailor",
                                     description="Draft a resume tailored to an approved job (resume agent).")
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    out = report_logger()
    try:
        view = service.tailor(args.job_id)
    except ValueError as e:
        out.info(f"can't tailor: {e}")
        return 1
    folder = DATA_DIR / "resumes"
    folder.mkdir(parents=True, exist_ok=True)
    md, docx = folder / f"{view.filename}.md", folder / f"{view.filename}.docx"
    md.write_text(view.markdown, encoding="utf-8")
    docx.write_bytes(service.resume_docx(args.job_id))
    out.info(view.markdown)
    if view.notes:
        out.info("Notes (not on the resume):")
        for note in view.notes:
            out.info(f"  - {note}")
    out.info(f"\nsaved {md} and {docx} (by {view.model})")
    return 0


def ui_cmd(argv: list[str]) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.address=localhost",          # this machine only
               "--browser.gatherUsageStats=false",
               "--theme.primaryColor=#24476b", *argv]
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging()
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        return usage(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        return discover_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        return analyze_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        return review_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        return ui_cmd(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "tailor":
        return tailor_cmd(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="a single job posting URL")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.url))
    except KeyboardInterrupt:
        report_logger().info("\nstopped; the job keeps its current status and will ask again next time")
        return 130


if __name__ == "__main__":
    sys.exit(main())
