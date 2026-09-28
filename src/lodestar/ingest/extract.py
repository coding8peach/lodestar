"""Build a JobPosting from a page's schema.org JSON-LD."""

import html as html_lib
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from bs4 import BeautifulSoup, NavigableString

from lodestar.ingest.jsonld import find_job_posting
from lodestar.schemas.job import ClassificationLayer, JobPosting, Source, job_id_from_url

log = logging.getLogger(__name__)

# Paragraph-level blocks get a blank line around them; list items and table rows a single line break.
PARAGRAPH_TAGS = ["p", "div", "section", "article", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table"]
LINE_TAGS = ["li", "tr"]


class ExtractionError(ValueError):
    """The page can't be turned into a JobPosting."""


def html_to_text(fragment: str) -> str:
    """HTML description -> readable plain text, keeping paragraph and list structure."""
    if "&lt;" in fragment:  # some ATSs HTML-escape the description inside JSON-LD
        fragment = html_lib.unescape(fragment)
    soup = BeautifulSoup(fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, NavigableString("- "))
    for tag in soup.find_all(PARAGRAPH_TAGS):
        tag.insert_before(NavigableString("\n\n"))
        tag.append(NavigableString("\n\n"))
    for tag in soup.find_all(LINE_TAGS):
        tag.insert_before(NavigableString("\n"))
    text = soup.get_text().replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _name(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _location(value: Any) -> str | None:
    places = value if isinstance(value, list) else [value]
    out: list[str] = []
    for place in places:
        if not isinstance(place, dict):
            continue
        addr = place.get("address", place)
        if isinstance(addr, str):
            text = addr.strip()
        elif isinstance(addr, dict):
            bits = [addr.get("addressLocality"), addr.get("addressRegion"), _name(addr.get("addressCountry"))]
            text = ", ".join(b.strip() for b in bits if isinstance(b, str) and b.strip())
        else:
            continue
        if text and text not in out:
            out.append(text)
    return "; ".join(out) or None


def _is_remote(value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and v.upper() == "TELECOMMUTE" for v in values)


def _num(value: Any) -> str | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return f"{n:,.0f}" if n == int(n) else f"{n:,.2f}"


def _salary(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    currency = value.get("currency") or ""
    inner = value.get("value")
    if isinstance(inner, dict):
        lo, hi = _num(inner.get("minValue")), _num(inner.get("maxValue"))
        single = _num(inner.get("value"))
        unit = inner.get("unitText") or ""
    else:
        lo = hi = None
        single = _num(inner)
        unit = value.get("unitText") or ""
    amount = f"{lo}–{hi}" if lo and hi else (lo or hi or single)
    if not amount:
        return None
    text = f"{currency} {amount}".strip()
    return f"{text} / {unit}" if unit else text


def _date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        log.warning("unparseable date: %r", value)
        return None


def extract_job(url: str, html: str, source: Source, classified_by: ClassificationLayer) -> JobPosting:
    data = find_job_posting(html)
    if data is None:
        raise ExtractionError(
            "no JSON-LD JobPosting in the page (it may load its content with JavaScript); "
            "LLM extraction arrives in step 4"
        )
    title = _name(data.get("title"))
    if not title:
        raise ExtractionError("JSON-LD JobPosting has no title")
    description = html_to_text(data.get("description") or "")
    if not description:
        raise ExtractionError("JSON-LD JobPosting has no description")

    return JobPosting(
        id=job_id_from_url(url),
        url=url,
        source=source,
        classified_by=classified_by,
        title=title,
        company=_name(data.get("hiringOrganization")),
        location=_location(data.get("jobLocation")),
        work_mode="remote" if _is_remote(data.get("jobLocationType")) else None,
        salary_text=_salary(data.get("baseSalary")),
        date_posted=_date(data.get("datePosted")),
        valid_through=_date(data.get("validThrough")),
        description=description,
        fetched_at=datetime.now(timezone.utc),
    )
