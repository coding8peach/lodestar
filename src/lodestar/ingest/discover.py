"""List watched companies' job boards, dedupe, and pre-filter. No LLM calls.

    boards (config/watchlist.yaml)
      -> one listing request per board (retries on 429 / 5xx / timeouts)
      -> JobPosting per job (same mapping as ingest_url; canonical hosted URL, so ids
         match jobs pasted by hand)
      -> already seen? (discoveries or jobs) -> skip
      -> validate + pre-filter (config/filters.yaml)
      -> passing jobs saved as queued; every new posting logged in discoveries
"""

import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
import yaml

from lodestar.db import get_job, save_job
from lodestar.db.status import now_utc
from lodestar.ingest.ats import ashby_board_url, map_ashby, map_greenhouse, map_lever
from lodestar.ingest.extract import ExtractionError
from lodestar.ingest.fetch import USER_AGENT
from lodestar.ingest.prefilter import FilterRules, prefilter
from lodestar.ingest.validate import validate_job
from lodestar.paths import PROJECT_ROOT
from lodestar.schemas.job import JobPosting, job_id_from_url

log = logging.getLogger(__name__)

WATCHLIST_FILE = PROJECT_ROOT / "config" / "watchlist.yaml"
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAYS = (1.0, 3.0)      # seconds before the 2nd and 3rd attempts
PAUSE_BETWEEN_BOARDS = 0.5


@dataclass(frozen=True)
class Board:
    company: str
    ats: Literal["greenhouse", "lever", "ashby"]
    board: str


@dataclass
class BoardReport:
    board: Board
    listed: int = 0
    seen: int = 0
    queued: list[JobPosting] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    error: str | None = None


class BoardError(RuntimeError):
    pass


def load_watchlist(path: Path = WATCHLIST_FILE) -> list[Board]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    blocked = {(b["ats"], b["board"]) for b in cfg.get("blocklist") or []}
    boards = []
    for entry in cfg.get("companies") or []:
        board = Board(company=entry["company"], ats=entry["ats"], board=entry["board"])
        if board.ats not in ("greenhouse", "lever", "ashby"):
            raise ValueError(f"{board.company}: unknown ats {board.ats!r}")
        if (board.ats, board.board) in blocked:
            log.warning("skipping %s/%s: on the blocklist", board.ats, board.board)
            continue
        boards.append(board)
    return boards


def _get(client: httpx.Client, url: str):
    """GET JSON with retries on rate limits, server errors, timeouts and dropped connections."""
    last: Exception | None = None
    for attempt, delay in enumerate((0.0, *RETRY_DELAYS)):
        if delay:
            time.sleep(delay)
        try:
            resp = client.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            continue
        if resp.status_code == 404:
            raise BoardError("board not found (check the board name)")
        if resp.status_code in RETRY_STATUS:
            last = BoardError(f"HTTP {resp.status_code}")
            continue
        if resp.status_code >= 400:
            raise BoardError(f"HTTP {resp.status_code}")
        return resp.json()
    raise BoardError(f"gave up after {len(RETRY_DELAYS) + 1} attempts: {last}")


def list_board(client: httpx.Client, board: Board) -> list[tuple[str, dict]]:
    """(canonical posting URL, JobPosting fields) for every open job on the board."""
    if board.ats == "greenhouse":
        data = _get(client, f"https://boards-api.greenhouse.io/v1/boards/{board.board}/jobs?content=true")
        return [(f"https://job-boards.greenhouse.io/{board.board}/jobs/{j['id']}", map_greenhouse(j))
                for j in data.get("jobs") or []]
    if board.ats == "lever":
        data = _get(client, f"https://api.lever.co/v0/postings/{board.board}?mode=json")
        return [(f"https://jobs.lever.co/{board.board}/{j['id']}", map_lever(j)) for j in data or []]
    data = _get(client, ashby_board_url(board.board))
    return [(f"https://jobs.ashbyhq.com/{board.board}/{j['id']}", map_ashby(j))
            for j in data.get("jobs") or [] if j.get("isListed", True)]


def _record(conn, job_id, url, board, fields, verdict, reason):
    now = now_utc()
    with conn:
        conn.execute(
            "INSERT INTO discoveries (job_id, url, company, ats, board, title, location, work_mode, verdict,"
            " reason, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(job_id) DO UPDATE SET verdict = excluded.verdict, reason = excluded.reason,"
            " last_seen_at = excluded.last_seen_at",
            (job_id, url, fields.get("company") or board.company, board.ats, board.board, fields.get("title") or "?",
             fields.get("location"), fields.get("work_mode"), verdict, reason, now, now),
        )


def discover_board(
    conn: sqlite3.Connection, client: httpx.Client, board: Board, rules: FilterRules,
    dry_run: bool = False, recheck: bool = False,
) -> BoardReport:
    report = BoardReport(board)
    try:
        postings = list_board(client, board)
    except (BoardError, ValueError, KeyError) as e:
        report.error = str(e)
        return report
    report.listed = len(postings)
    fetched_at = datetime.now(timezone.utc)
    for url, fields in postings:
        job_id = job_id_from_url(url)
        seen = conn.execute("SELECT verdict FROM discoveries WHERE job_id = ?", (job_id,)).fetchone()
        known = get_job(conn, job_id)
        if known is not None:
            report.seen += 1
            if not dry_run and not known.company:
                # e.g. an Ashby/Lever job pasted before company lookup existed
                with conn:
                    conn.execute("UPDATE jobs SET company = ? WHERE id = ? AND company IS NULL",
                                 (board.company, job_id))
            if not dry_run and seen is None:
                _record(conn, job_id, url, board, fields, "known", "already in jobs")
            elif not dry_run:
                with conn:
                    conn.execute("UPDATE discoveries SET last_seen_at = ? WHERE job_id = ?", (now_utc(), job_id))
            continue
        if seen is not None and not recheck:
            report.seen += 1
            if not dry_run:
                with conn:
                    conn.execute("UPDATE discoveries SET last_seen_at = ? WHERE job_id = ?", (now_utc(), job_id))
            continue
        try:
            job = validate_job(JobPosting(
                id=job_id, url=url, source=board.ats, classified_by="url_rule", fetched_at=fetched_at,
                **{**fields, "company": fields.get("company") or board.company}))
        except (ExtractionError, ValueError) as e:
            report.skipped["invalid posting"] += 1
            if not dry_run:
                _record(conn, job_id, url, board, fields, "invalid", str(e)[:200])
            continue
        passes, reason = prefilter(job, rules)
        if not passes:
            report.skipped[reason if not reason.startswith("location: ") or "outside" in reason
                           else "location: elsewhere"] += 1
            if not dry_run:
                _record(conn, job_id, url, board, fields, "skipped", reason)
            continue
        report.queued.append(job)
        if not dry_run:
            save_job(conn, job)
            _record(conn, job_id, url, board, fields, "queued", reason)
    return report


def discover(
    conn: sqlite3.Connection, boards: list[Board], rules: FilterRules,
    dry_run: bool = False, recheck: bool = False, client: httpx.Client | None = None,
) -> list[BoardReport]:
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20.0, follow_redirects=True)
    try:
        reports = []
        for i, board in enumerate(boards):
            if i:
                time.sleep(PAUSE_BETWEEN_BOARDS)
            report = discover_board(conn, client, board, rules, dry_run, recheck)
            if report.error:
                log.warning("%s (%s/%s): %s", board.company, board.ats, board.board, report.error)
            reports.append(report)
        return reports
    finally:
        if own:
            client.close()
