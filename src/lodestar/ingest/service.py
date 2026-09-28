"""ingest_url(): the one ingestion entry point exposed over MCP."""

import logging
import sqlite3
from typing import Literal

from pydantic import BaseModel

from lodestar.db import get_job, save_job
from lodestar.ingest.classify import strip_apply_suffix
from lodestar.ingest.pipeline import build_posting
from lodestar.ingest.validate import validate_job
from lodestar.schemas.job import Source, job_id_from_url, normalize_url

log = logging.getLogger(__name__)


class IngestResult(BaseModel):
    job_id: str
    status: Literal["saved", "already_saved"]
    url: str
    title: str
    company: str | None


def ingest_url(conn: sqlite3.Connection, url: str, source: Source = "pasted_url") -> IngestResult:
    """Fetch, classify, extract, validate and save one posting.

    A URL that's already saved returns the stored job without refetching.
    Raises ExtractionError, httpx.HTTPError, or ValueError (not an http(s) URL).
    """
    url = strip_apply_suffix(url)
    job_id = job_id_from_url(url)
    existing = get_job(conn, job_id)
    if existing is not None:
        log.info("already saved: %s", job_id)
        return IngestResult(job_id=job_id, status="already_saved", url=existing.url,
                            title=existing.title, company=existing.company)

    job = validate_job(build_posting(url, source=source))
    save_job(conn, job)
    return IngestResult(job_id=job.id, status="saved", url=normalize_url(url), title=job.title, company=job.company)
