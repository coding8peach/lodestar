"""Ingestion: plain, individually testable functions. Only ingest_url() is exposed over MCP."""

from lodestar.ingest.ats import company_from_page_title, fetch_from_ats
from lodestar.ingest.classify import AtsMatch, classify_page, match_ats, strip_apply_suffix
from lodestar.ingest.extract import ExtractionError, extract_job
from lodestar.ingest.fetch import fetch_page
from lodestar.ingest.pipeline import build_posting
from lodestar.ingest.service import IngestResult, ingest_url
from lodestar.ingest.validate import validate_job

__all__ = [
    "AtsMatch", "ExtractionError", "IngestResult", "build_posting", "classify_page",
    "company_from_page_title", "extract_job", "fetch_from_ats", "fetch_page", "ingest_url",
    "match_ats", "strip_apply_suffix", "validate_job",
]
