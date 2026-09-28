"""Decide whether a page is a single job posting.

Deterministic layers first, in order (see CLAUDE.md):
  1. url_rule: known ATS posting URL shapes (these are then fetched through the ATS API)
  2. json_ld:  a schema.org JobPosting in the page
  3. llm:      fallback, added in build step 4
Returns the deciding layer, or None when no layer decides (treat as not a posting for now).
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from lodestar.ingest.jsonld import find_job_posting
from lodestar.schemas.job import ClassificationLayer

log = logging.getLogger(__name__)

Ats = Literal["greenhouse", "lever", "ashby"]

_UUID = r"[0-9a-fA-F-]{36}"
_GREENHOUSE = re.compile(r"^/(?P<board>[^/]+)/jobs/(?P<job>\d+)/?$")
_LEVER = re.compile(rf"^/(?P<board>[^/]+)/(?P<job>{_UUID})/?$")
_ASHBY = re.compile(rf"^/(?P<board>[^/]+)/(?P<job>{_UUID})/?$")

# host -> (ATS, path pattern of a single posting). Board index pages
# (e.g. jobs.lever.co/acme) and /apply pages deliberately don't match.
POSTING_URL_RULES: dict[str, tuple[Ats, re.Pattern[str]]] = {
    "boards.greenhouse.io": ("greenhouse", _GREENHOUSE),
    "job-boards.greenhouse.io": ("greenhouse", _GREENHOUSE),
    "jobs.lever.co": ("lever", _LEVER),
    "jobs.eu.lever.co": ("lever", _LEVER),
    "jobs.ashbyhq.com": ("ashby", _ASHBY),
}


@dataclass(frozen=True)
class AtsMatch:
    ats: Ats
    host: str
    board: str   # Greenhouse board token / Lever site / Ashby org
    job: str     # the ATS's own job id


# Application-form pages that sit under a posting URL.
_APPLY_SUFFIXES = {
    "jobs.ashbyhq.com": "/application",
    "jobs.lever.co": "/apply",
    "jobs.eu.lever.co": "/apply",
}


def strip_apply_suffix(url: str) -> str:
    """Turn an ATS application-form URL back into its posting URL; other URLs pass through."""
    parts = urlsplit(url.strip())
    suffix = _APPLY_SUFFIXES.get(parts.netloc.lower())
    path = parts.path.rstrip("/")
    if suffix and path.endswith(suffix):
        return urlunsplit(parts._replace(path=path[: -len(suffix)]))
    return url.strip()


def match_ats(url: str) -> AtsMatch | None:
    """If `url` is a single posting on a known ATS, say which one and its board/job ids."""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    rule = POSTING_URL_RULES.get(host)
    if not rule:
        return None
    ats, pattern = rule
    m = pattern.match(parts.path)
    return AtsMatch(ats, host, m["board"], m["job"]) if m else None


def classify_page(url: str, html: str) -> ClassificationLayer | None:
    if match_ats(url):
        log.info("classified as posting by url_rule: %s", url)
        return "url_rule"
    if find_job_posting(html) is not None:
        log.info("classified as posting by json_ld: %s", url)
        return "json_ld"
    log.info("not classified (no url rule, no JSON-LD JobPosting): %s", url)
    return None
