"""Checks and cleanup between extraction and saving."""

import logging
from datetime import date

from lodestar.ingest.extract import ExtractionError
from lodestar.schemas.job import JobPosting

log = logging.getLogger(__name__)

# Shorter than this is almost certainly a stub or a listing, not a full posting.
MIN_DESCRIPTION_CHARS = 200


def _one_line(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None


def validate_job(job: JobPosting) -> JobPosting:
    """Trim stray whitespace and reject postings too thin to analyze. Returns a cleaned copy."""
    title = _one_line(job.title)
    if not title:
        raise ExtractionError("posting has an empty title")
    description = job.description.strip()
    if len(description) < MIN_DESCRIPTION_CHARS:
        raise ExtractionError(
            f"description is only {len(description)} characters; this doesn't look like a full posting"
        )
    if job.valid_through and job.valid_through < date.today():
        log.warning("job %s: valid_through %s has passed; the posting may be closed", job.id, job.valid_through)
    return job.model_copy(update={
        "title": title,
        "company": _one_line(job.company),
        "location": _one_line(job.location),
        "salary_text": _one_line(job.salary_text),
        "description": description,
    })
