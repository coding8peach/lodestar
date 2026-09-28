"""Fetch postings on known ATSs through their public job-board APIs (no key needed).

ATS pages don't reliably include JSON-LD, and their HTML changes without notice;
the APIs return clean JSON.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from lodestar.ingest.classify import AtsMatch
from lodestar.ingest.extract import ExtractionError, _date, _num, html_to_text
from lodestar.ingest.fetch import USER_AGENT, fetch_page
from lodestar.schemas.common import WorkMode
from lodestar.schemas.job import JobPosting, Source, job_id_from_url

log = logging.getLogger(__name__)


def _get_json(url: str, timeout: float = 20.0) -> Any:
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    log.info("fetched %s", url)
    return resp.json()


def _work_mode(value: Any) -> WorkMode | None:
    if not isinstance(value, str):
        return None
    v = "".join(c for c in value.lower() if c.isalpha())  # "OnSite", "on-site" -> "onsite"
    return v if v in ("remote", "hybrid", "onsite") else None


def _range(currency: Any, lo: Any, hi: Any, unit: Any = None) -> str | None:
    lo_s, hi_s = _num(lo), _num(hi)
    amount = f"{lo_s}–{hi_s}" if lo_s and hi_s and lo_s != hi_s else (lo_s or hi_s)
    if not amount:
        return None
    text = f"{currency or ''} {amount}".strip()
    return f"{text} / {unit}" if unit else text


def _join(*parts: str | None) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


# --- mapping: one ATS job object -> JobPosting fields --------------------------------
# Shared by single-posting fetches (ingest_url) and board listings (discover).

def map_greenhouse(data: dict) -> dict:
    ranges = [
        _range(r.get("currency_type"),
               r["min_cents"] / 100 if r.get("min_cents") is not None else None,
               r["max_cents"] / 100 if r.get("max_cents") is not None else None)
        for r in data.get("pay_input_ranges") or []
    ]
    return {
        "title": data.get("title"),
        "company": data.get("company_name"),
        "location": (data.get("location") or {}).get("name"),
        "work_mode": None,  # Greenhouse has no remote flag
        "salary_text": "; ".join(r for r in ranges if r) or None,
        "date_posted": _date(data.get("first_published") or data.get("updated_at")),
        "description": html_to_text(data.get("content") or ""),
    }


def map_lever(data: dict) -> dict:
    categories = data.get("categories") or {}
    lists = [
        _join(item.get("text"), html_to_text(item.get("content") or ""))
        for item in data.get("lists") or []
    ]
    created = data.get("createdAt")
    salary = data.get("salaryRange") or {}
    return {
        "title": data.get("text"),
        "company": None,  # not in Lever's API; filled from the page title or the watchlist
        "location": categories.get("location"),
        "work_mode": _work_mode(data.get("workplaceType")),
        "salary_text": _range(salary.get("currency"), salary.get("min"), salary.get("max"), salary.get("interval")),
        "date_posted": datetime.fromtimestamp(created / 1000, timezone.utc).date() if created else None,
        "description": _join(
            data.get("descriptionPlain") or html_to_text(data.get("description") or ""),
            *lists,
            data.get("additionalPlain") or html_to_text(data.get("additional") or ""),
        ),
    }


def map_ashby(job: dict) -> dict:
    comp = job.get("compensation") or {}
    work_mode = _work_mode(job.get("workplaceType")) or ("remote" if job.get("isRemote") else None)
    return {
        "title": job.get("title"),
        "company": None,  # not in Ashby's API; filled from the page title or the watchlist
        "location": job.get("location"),
        "work_mode": work_mode,
        "salary_text": comp.get("scrapeableCompensationSalarySummary") or comp.get("compensationTierSummary"),
        "date_posted": _date(job.get("publishedAt")),
        "description": job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml") or ""),
    }


# --- endpoints ----------------------------------------------------------------------

def lever_api_host(posting_host: str) -> str:
    return "api.eu.lever.co" if posting_host == "jobs.eu.lever.co" else "api.lever.co"


def ashby_board_url(board: str) -> str:
    return f"https://api.ashbyhq.com/posting-api/job-board/{board}?includeCompensation=true"


def _greenhouse(m: AtsMatch) -> dict:
    return map_greenhouse(
        _get_json(f"https://boards-api.greenhouse.io/v1/boards/{m.board}/jobs/{m.job}?pay_transparency=true"))


def _lever(m: AtsMatch) -> dict:
    return map_lever(_get_json(f"https://{lever_api_host(m.host)}/v0/postings/{m.board}/{m.job}"))


def _ashby(m: AtsMatch) -> dict:
    data = _get_json(ashby_board_url(m.board))
    job = next((j for j in data.get("jobs") or [] if str(j.get("id", "")).lower() == m.job.lower()), None)
    if job is None:
        raise ExtractionError(f"job {m.job} not found on Ashby board {m.board!r} (it may be closed)")
    return map_ashby(job)


def company_from_page_title(ats: str, page_title: str, job_title: str) -> str | None:
    """Company name from a hosted posting page's <title>, anchored on the job title.

    Ashby: "<job title> @ <company>".  Lever: "<company> - <job title>".
    Returns None when the title doesn't have the expected shape.
    """
    t, jt = " ".join(page_title.split()), " ".join(job_title.split())
    if ats == "ashby":
        if t.startswith(f"{jt} @ "):
            return t[len(jt) + 3 :].strip() or None
        if " @ " in t:
            return t.rsplit(" @ ", 1)[1].strip() or None
    if ats == "lever" and t.endswith(f" - {jt}"):
        return t[: -len(jt) - 3].strip() or None
    return None


def _company_from_page(url: str, m: AtsMatch, job_title: str) -> str | None:
    """Fetch the hosted page for its <title>. Never fails ingestion: returns None on any problem."""
    try:
        html = fetch_page(url)
    except httpx.HTTPError as e:
        log.warning("couldn't fetch %s for the company name: %s", url, e)
        return None
    tag = BeautifulSoup(html, "html.parser").title
    page_title = tag.get_text() if tag else ""
    company = company_from_page_title(m.ats, page_title, job_title)
    if company is None:
        log.warning("no company name in page title %r", page_title)
    return company


_ADAPTERS = {"greenhouse": _greenhouse, "lever": _lever, "ashby": _ashby}


def fetch_from_ats(url: str, m: AtsMatch, source: Source) -> JobPosting:
    """Build a JobPosting for an ATS posting URL via that ATS's public API."""
    fields = _ADAPTERS[m.ats](m)
    if not fields.get("title"):
        raise ExtractionError(f"{m.ats} API returned no title")
    if not fields.get("company") and m.ats in ("ashby", "lever"):
        fields["company"] = _company_from_page(url, m, fields["title"])
    if not fields.get("description"):
        raise ExtractionError(f"{m.ats} API returned no description")
    return JobPosting(
        id=job_id_from_url(url),
        url=url,
        source=source,
        classified_by="url_rule",
        fetched_at=datetime.now(timezone.utc),
        **fields,
    )
