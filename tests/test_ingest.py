import json
from pathlib import Path

import pytest

from lodestar.ingest import ExtractionError, classify_page, extract_job
from lodestar.ingest.extract import html_to_text
from lodestar.ingest.jsonld import find_job_posting
from lodestar.schemas import job_id_from_url

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GREENHOUSE_HTML = (FIXTURES / "greenhouse_posting.html").read_text(encoding="utf-8")
PLAIN_HTML = "<html><body><h1>Jobs</h1><a href='/1'>Job 1</a><a href='/2'>Job 2</a></body></html>"
UUID = "3f2a9c1e-1234-4abc-9def-0123456789ab"


def jsonld_page(obj) -> str:
    return f'<html><head><script type="application/ld+json">{json.dumps(obj)}</script></head></html>'


# --- classify -------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        f"https://jobs.lever.co/acme/{UUID}",
        "https://boards.greenhouse.io/acme/jobs/4012345",
        "https://job-boards.greenhouse.io/acme/jobs/4012345",
        f"https://jobs.ashbyhq.com/acme/{UUID}",
    ],
)
def test_url_rule_postings(url):
    assert classify_page(url, PLAIN_HTML) == "url_rule"


@pytest.mark.parametrize(
    "url",
    [
        "https://jobs.lever.co/acme",                       # board index, not a posting
        "https://boards.greenhouse.io/acme",
        f"https://jobs.lever.co/acme/{UUID}/apply",          # application form
        "https://www.indeed.com/q-backend-engineer-jobs.html",  # aggregator
    ],
)
def test_non_postings_not_classified(url):
    assert classify_page(url, PLAIN_HTML) is None


def test_json_ld_layer_on_company_site():
    assert classify_page("https://careers.acme.com/jobs/123", GREENHOUSE_HTML) == "json_ld"


def test_url_rule_wins_over_json_ld():
    assert classify_page("https://boards.greenhouse.io/acme/jobs/1", GREENHOUSE_HTML) == "url_rule"


# --- JSON-LD discovery ----------------------------------------------------

def test_finds_posting_in_graph_and_type_list():
    page = jsonld_page({"@context": "https://schema.org", "@graph": [
        {"@type": "Organization", "name": "Acme"},
        {"@type": ["JobPosting"], "title": "Engineer"},
    ]})
    assert find_job_posting(page)["title"] == "Engineer"


def test_skips_malformed_block():
    page = '<script type="application/ld+json">{not json</script>' + jsonld_page({"@type": "JobPosting", "title": "X"})
    assert find_job_posting(page)["title"] == "X"


# --- extract --------------------------------------------------------------

def test_extract_full_posting():
    url = "https://boards.greenhouse.io/acme/jobs/1?gh_src=abc"
    job = extract_job(url, GREENHOUSE_HTML, source="pasted_url", classified_by="url_rule")
    assert job.id == job_id_from_url(url)
    assert job.url == "https://boards.greenhouse.io/acme/jobs/1"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme"
    assert job.location == "San Francisco, CA, US; New York, NY, US"
    assert job.work_mode is None
    assert job.seniority is None
    assert job.salary_text == "USD 150,000–190,000 / YEAR"
    assert str(job.date_posted) == "2026-09-01"
    assert str(job.valid_through) == "2026-12-01"
    assert job.description == "We build agent tools.\n\nRequirements\n\n- 5+ years of Python\n- PostgreSQL experience"


def test_extract_remote_and_missing_fields():
    page = jsonld_page({"@type": "JobPosting", "title": "Engineer", "description": "<p>Hi</p>",
                        "jobLocationType": "TELECOMMUTE"})
    job = extract_job("https://careers.acme.com/1", page, source="pasted_url", classified_by="json_ld")
    assert job.work_mode == "remote"
    assert job.company is None and job.location is None and job.salary_text is None


def test_extract_without_json_ld_fails():
    with pytest.raises(ExtractionError, match="no JSON-LD"):
        extract_job(f"https://jobs.lever.co/acme/{UUID}", PLAIN_HTML, source="pasted_url", classified_by="url_rule")


def test_extract_without_description_fails():
    page = jsonld_page({"@type": "JobPosting", "title": "Engineer"})
    with pytest.raises(ExtractionError, match="no description"):
        extract_job("https://careers.acme.com/1", page, source="pasted_url", classified_by="json_ld")


def test_html_to_text_line_breaks():
    assert html_to_text("Line one<br>Line two<p>Para</p>") == "Line one\nLine two\n\nPara"
