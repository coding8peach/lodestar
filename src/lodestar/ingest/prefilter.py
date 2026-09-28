"""Deterministic pre-filter: decide cheaply whether a posting is worth an LLM analysis.

No LLM here: an agent should never be needed to learn that "Senior iOS Engineer — London"
isn't relevant. Rules live in config/filters.yaml. Every decision comes with a reason.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from lodestar.paths import PROJECT_ROOT
from lodestar.schemas.job import JobPosting

FILTERS_FILE = PROJECT_ROOT / "config" / "filters.yaml"


def _words(terms: list[str]) -> re.Pattern[str]:
    """One pattern matching any term as a whole word (so "intern" doesn't match "internal")."""
    return re.compile(r"(?<![\w.])(" + "|".join(re.escape(t.lower()) for t in terms) + r")(?![\w])", re.I)


@dataclass(frozen=True)
class FilterRules:
    include_title: re.Pattern[str]
    exclude_title: re.Pattern[str]
    exclude_title_patterns: tuple[re.Pattern[str], ...]
    bay_area: re.Pattern[str]
    us_markers: re.Pattern[str]
    non_us_markers: re.Pattern[str]


def load_rules(path: Path = FILTERS_FILE) -> FilterRules:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    title, loc = cfg["title"], cfg["location"]
    return FilterRules(
        include_title=_words(title["include_any"]),
        exclude_title=_words(title["exclude_any"]),
        exclude_title_patterns=tuple(re.compile(p, re.I) for p in title.get("exclude_patterns", [])),
        bay_area=_words(loc["bay_area"]),
        us_markers=_words(loc["us_markers"]),
        non_us_markers=_words(loc["non_us_markers"]),
    )


def prefilter(job: JobPosting, rules: FilterRules) -> tuple[bool, str]:
    """(passes, reason). Reasons are short and stable, so they can be counted."""
    title = job.title
    if not rules.include_title.search(title):
        return False, "title: not an engineering role"
    if m := rules.exclude_title.search(title):
        return False, f"title: {m.group(1).lower()}"
    for pattern in rules.exclude_title_patterns:
        if pattern.search(title):
            return False, "title: below senior"

    location = job.location or ""
    remote = job.work_mode == "remote" or re.search(r"\bremote\b", location, re.I) is not None
    if rules.bay_area.search(location):
        return True, "bay area"
    if remote:
        if rules.non_us_markers.search(location) and not rules.us_markers.search(location):
            return False, "location: remote outside the US"
        return True, "remote"
    if not location.strip():
        return True, "no location stated"  # the fit agent's hard-constraint check decides
    return False, f"location: {location[:40]}"
