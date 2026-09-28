"""Discovery: board listings (stubbed HTTP), dedupe, pre-filter, recording. No network, no LLM."""

import json
import re
from datetime import datetime, timezone

import httpx
import pytest

from lodestar.db import connect, get_status, init_db, save_job
from lodestar.ingest import discover as discover_module
from lodestar.ingest.discover import Board, discover, list_board, load_watchlist
from lodestar.ingest.prefilter import load_rules, prefilter
from lodestar.schemas import JobPosting, job_id_from_url

BODY = "We build backend systems in Python and PostgreSQL. " * 10
UUID = "3f2a9c1e-1234-4abc-9def-0123456789a"  # + one hex digit per job


def ashby_job(n: int, title: str, location: str, remote=False, listed=True) -> dict:
    return {"id": f"{UUID}{n}", "title": title, "location": location, "isRemote": remote,
            "workplaceType": "Remote" if remote else "Hybrid", "publishedAt": "2026-09-01",
            "descriptionPlain": BODY, "isListed": listed}


ASHBY = {"jobs": [
    ashby_job(1, "Senior Backend Engineer", "San Francisco"),
    ashby_job(2, "Senior iOS Engineer", "San Francisco"),
    ashby_job(3, "Account Executive", "San Francisco"),
    ashby_job(4, "Staff Engineer, Platform", "Remote - Europe", remote=True),
    ashby_job(5, "Software Engineer II", "Remote - US", remote=True),
    ashby_job(6, "Senior Software Engineer", "New York, NY"),
    ashby_job(7, "Staff Software Engineer", "Remote - US", remote=True),
    ashby_job(8, "Senior Engineer (unlisted)", "San Francisco", listed=False),
]}
GREENHOUSE = {"jobs": [
    {"id": 101, "title": "Senior Backend Engineer (Python)", "location": {"name": "Remote, US"},
     "content": "&lt;p&gt;" + BODY + "&lt;/p&gt;", "company_name": "Oura", "first_published": "2026-09-02"},
    {"id": 102, "title": "Senior Engineer", "location": {"name": "Oakland, CA"}, "content": "&lt;p&gt;short&lt;/p&gt;"},
]}
LEVER = [{"id": "abc-1", "text": "Senior Full-Stack Engineer", "categories": {"location": "Palo Alto, CA"},
          "workplaceType": "hybrid", "createdAt": 1788000000000, "descriptionPlain": BODY}]


def transport(fail: set[str] = frozenset(), calls: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if calls is not None:
            calls.append(url)
        if any(f in url for f in fail):
            return httpx.Response(503)
        if "api.ashbyhq.com" in url and "/acme" in url:
            return httpx.Response(200, json=ASHBY)
        if "boards-api.greenhouse.io" in url and "/oura/" in url:
            return httpx.Response(200, json=GREENHOUSE)
        if "api.lever.co" in url and "/leverco" in url:
            return httpx.Response(200, json=LEVER)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


BOARDS = [Board("Acme", "ashby", "acme"), Board("Oura", "greenhouse", "oura"), Board("LeverCo", "lever", "leverco")]


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("LODESTAR_DB", str(tmp_path / "t.sqlite"))
    monkeypatch.setattr(discover_module, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(discover_module, "PAUSE_BETWEEN_BOARDS", 0.0)
    c = connect()
    init_db(c)
    yield c
    c.close()


def run(conn, boards=BOARDS, **kw):
    with httpx.Client(transport=transport(kw.pop("fail", set()), kw.pop("calls", None))) as client:
        return {r.board.board: r for r in discover(conn, boards, load_rules(), client=client, **kw)}


# --- listing ------------------------------------------------------------------------

def test_listing_uses_canonical_hosted_urls():
    with httpx.Client(transport=transport()) as client:
        ashby = list_board(client, BOARDS[0])
        gh = list_board(client, BOARDS[1])
        lever = list_board(client, BOARDS[2])
    assert ashby[0][0] == f"https://jobs.ashbyhq.com/acme/{UUID}1"
    assert len(ashby) == 7  # the unlisted job is left out
    assert gh[0][0] == "https://job-boards.greenhouse.io/oura/jobs/101"
    assert lever[0][0] == "https://jobs.lever.co/leverco/abc-1"


# --- the whole pass -----------------------------------------------------------------

def test_discover_queues_passing_jobs_and_logs_the_rest(conn):
    reports = run(conn)
    acme = reports["acme"]
    assert acme.listed == 7
    assert [j.title for j in acme.queued] == ["Senior Backend Engineer", "Staff Software Engineer"]
    assert acme.skipped == {"title: ios": 1, "title: not an engineering role": 1,
                            "location: remote outside the US": 1, "title: below senior": 1,
                            "location: elsewhere": 1}
    queued = acme.queued[0]
    assert queued.company == "Acme"  # from the watchlist; Ashby's API has no company name
    assert get_status(conn, queued.id) == "queued"
    rows = dict(conn.execute("SELECT verdict, COUNT(*) FROM discoveries GROUP BY verdict").fetchall())
    assert rows == {"queued": 4, "skipped": 5, "invalid": 1}  # acme 2 + oura 1 + lever 1; oura 102 too short
    assert reports["oura"].queued[0].company == "Oura"
    assert reports["leverco"].queued[0].work_mode == "hybrid"


def test_second_run_skips_everything_already_seen(conn):
    run(conn)
    again = run(conn)
    assert all(r.seen == r.listed and not r.queued and not r.skipped for r in again.values())


def test_job_pasted_by_hand_is_known_not_requeued(conn):
    url = f"https://jobs.ashbyhq.com/acme/{UUID}1"
    save_job(conn, JobPosting(id=job_id_from_url(url), url=url, source="pasted_url", classified_by="url_rule",
                              title="Senior Backend Engineer", description=BODY,
                              fetched_at=datetime(2026, 9, 1, tzinfo=timezone.utc)))
    acme = run(conn)["acme"]
    assert acme.seen == 1 and "Senior Backend Engineer" not in [j.title for j in acme.queued]
    verdict = conn.execute("SELECT verdict FROM discoveries WHERE job_id = ?", (job_id_from_url(url),)).fetchone()[0]
    assert verdict == "known"


def test_dry_run_saves_nothing(conn):
    reports = run(conn, dry_run=True)
    assert len(reports["acme"].queued) == 2
    assert conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_recheck_refilters_skipped_postings(conn):
    run(conn)
    reports = run(conn, recheck=True)
    assert reports["acme"].skipped["title: ios"] == 1  # filtered again instead of counted as seen


def test_failing_board_does_not_stop_the_others(conn):
    calls: list[str] = []
    reports = run(conn, fail={"api.ashbyhq.com"}, calls=calls)
    assert "gave up after 3 attempts" in reports["acme"].error
    assert sum("api.ashbyhq.com" in c for c in calls) == 3  # retried
    assert len(reports["oura"].queued) == 1 and len(reports["leverco"].queued) == 1


def test_unknown_board_is_reported():
    with httpx.Client(transport=transport()) as client:
        with pytest.raises(Exception, match="board not found"):
            list_board(client, Board("Nope", "ashby", "nope"))


# --- configuration ------------------------------------------------------------------

def test_watchlist_blocklist(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(json.dumps({"companies": [{"company": "A", "ats": "ashby", "board": "a"},
                                              {"company": "J", "ats": "lever", "board": "jobgether"}],
                                "blocklist": [{"ats": "lever", "board": "jobgether"}]}))
    assert [b.board for b in load_watchlist(path)] == ["a"]


def test_shipped_watchlist_loads():
    boards = load_watchlist()
    assert boards and all(re.fullmatch(r"[a-z0-9-]+", b.board) for b in boards)


def job(title, location=None, work_mode=None) -> JobPosting:
    url = "https://jobs.ashbyhq.com/x/00000000-0000-0000-0000-000000000000"
    return JobPosting(id=job_id_from_url(url), url=url, source="ashby", classified_by="url_rule", title=title,
                      location=location, work_mode=work_mode, description=BODY,
                      fetched_at=datetime(2026, 9, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize("title, location, work_mode, expected", [
    ("Senior Backend Engineer", "San Francisco, CA", None, (True, "bay area")),
    ("Internal Tools Engineer", "Remote - US", None, (True, "remote")),       # "intern" is a whole word only
    ("Software Engineer II", "SF", None, (False, "title: below senior")),
    ("Senior iOS Engineer", "London", None, (False, "title: ios")),
    ("Engineering Manager", "SF", None, (False, "title: manager")),
    ("Principal Engineer", "SF", None, (False, "title: principal")),
    ("Staff Engineer", "Remote - Europe", None, (False, "location: remote outside the US")),
    ("Senior Software Engineer", "Toronto; Remote - US", None, (True, "remote")),
    ("Senior Engineer", "New York, NY", None, (False, "location: New York, NY")),
    ("Senior Engineer", "Anywhere", "remote", (True, "remote")),
    ("Senior Engineer", None, None, (True, "no location stated")),
    ("Member of Technical Staff", "Palo Alto", None, (True, "bay area")),
    ("Account Executive", "SF", None, (False, "title: not an engineering role")),
])
def test_prefilter_rules(title, location, work_mode, expected):
    assert prefilter(job(title, location, work_mode), load_rules()) == expected
