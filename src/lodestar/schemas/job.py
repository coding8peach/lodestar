"""A normalized job posting. Every ingestion source produces this same shape.

It records what the page said, not what Lodestar did with it: lifecycle status
lives on the `jobs` row. It holds no extracted requirements; the fit agent
extracts those.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lodestar.schemas.common import Seniority, WorkMode

Source = Literal["pasted_url", "serper", "greenhouse", "lever", "ashby"]
ClassificationLayer = Literal["url_rule", "json_ld", "llm"]

# Only parameters known to be tracking noise. Everything else is kept, because
# some career pages identify the job in the query string (e.g. ?gh_jid=123).
TRACKING_PARAMS = {"gh_src", "lever-source", "lever-origin", "gclid", "fbclid"}
TRACKING_PREFIXES = ("utm_",)


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k in TRACKING_PARAMS or k.startswith(TRACKING_PREFIXES)


def normalize_url(url: str) -> str:
    """Canonical form of a posting URL, so the same job always gets the same id."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"not an http(s) URL: {url!r}")
    path = parts.path.rstrip("/") or "/"
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def job_id_from_url(url: str) -> str:
    """Stable job id: a short hash of the normalized URL."""
    return "job_" + hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    url: str
    source: Source
    classified_by: ClassificationLayer
    title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    work_mode: WorkMode | None = None
    seniority: Seniority | None = None
    salary_text: str | None = None
    date_posted: date | None = None
    valid_through: date | None = None
    description: str = Field(min_length=1)  # full text, plain text
    fetched_at: datetime

    @field_validator("url")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_url(v)

    @field_validator("fetched_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _id_matches_url(self) -> JobPosting:
        expected = job_id_from_url(self.url)
        if self.id != expected:
            raise ValueError(f"id {self.id!r} does not match url (expected {expected!r})")
        return self
