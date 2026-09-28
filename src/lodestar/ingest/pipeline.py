"""URL -> JobPosting. This becomes the core of MCP ingest_url() in step 4 (plus validate + save)."""

import logging

from lodestar.ingest.ats import fetch_from_ats
from lodestar.ingest.classify import classify_page, match_ats, strip_apply_suffix
from lodestar.ingest.extract import ExtractionError, extract_job
from lodestar.ingest.fetch import fetch_page
from lodestar.schemas.job import JobPosting, Source

log = logging.getLogger(__name__)


def build_posting(url: str, source: Source = "pasted_url") -> JobPosting:
    """Known ATS URL -> its public API; anything else -> fetch the page and read its JSON-LD."""
    url = strip_apply_suffix(url)
    m = match_ats(url)
    if m:
        log.info("classified as posting by url_rule (%s board=%s job=%s)", m.ats, m.board, m.job)
        return fetch_from_ats(url, m, source)
    html = fetch_page(url)
    layer = classify_page(url, html)
    if layer is None:
        raise ExtractionError("not recognized as a single job posting (LLM fallback comes in step 4)")
    return extract_job(url, html, source=source, classified_by=layer)
