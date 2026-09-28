import httpx
import pytest

from lodestar.ingest import ExtractionError, build_posting, company_from_page_title, match_ats
from lodestar.ingest import ats as ats_module
from lodestar.ingest import pipeline as pipeline_module
from lodestar.schemas import job_id_from_url

UUID = "3f2a9c1e-1234-4abc-9def-0123456789ab"

GREENHOUSE_JSON = {
    "id": 8211828,
    "title": "Senior Software Engineer",
    "company_name": "Greenhouse",
    "location": {"name": "Ontario"},
    "first_published": "2026-09-10T12:00:00-04:00",
    "updated_at": "2026-09-20T12:00:00-04:00",
    "content": "&lt;p&gt;Build &lt;strong&gt;hiring&lt;/strong&gt; tools.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Ruby&lt;/li&gt;&lt;li&gt;React&lt;/li&gt;&lt;/ul&gt;",
    "pay_input_ranges": [{"min_cents": 12900000, "max_cents": 16450000, "currency_type": "CAD", "title": "National"}],
}

LEVER_JSON = {
    "id": UUID,
    "text": "Backend Engineer",
    "categories": {"location": "San Francisco, CA", "commitment": "Full-time"},
    "workplaceType": "hybrid",
    "createdAt": 1788000000000,
    "descriptionPlain": "We build things.",
    "lists": [{"text": "Requirements", "content": "<li>Python</li><li>Postgres</li>"}],
    "additionalPlain": "Benefits included.",
    "salaryRange": {"currency": "USD", "interval": "per-year-salary", "min": 150000, "max": 190000},
}

ASHBY_JSON = {
    "jobs": [
        {"id": "other-id", "title": "Other"},
        {
            "id": UUID,
            "title": "Staff Engineer",
            "location": "Remote - US",
            "workplaceType": "Remote",
            "isRemote": True,
            "publishedAt": "2026-09-01T00:00:00.000+00:00",
            "descriptionPlain": "Agents and data.",
            "compensation": {"scrapeableCompensationSalarySummary": "$180K - $220K"},
        },
    ]
}


@pytest.fixture
def api(monkeypatch):
    """Stub the HTTP layer: map URL substrings to canned JSON, record calls, forbid page fetches."""
    responses: dict[str, object] = {}
    pages: dict[str, str] = {}  # posting URL -> hosted page <title>
    calls: list[str] = []

    def fake_get_json(url, timeout=20.0):
        calls.append(url)
        for key, value in responses.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected API call: {url}")

    def no_page_fetch(url, timeout=20.0):
        raise AssertionError("ATS URLs must not go through the page-fetch/JSON-LD route")

    def fake_title_page(url, timeout=20.0):
        calls.append(url)
        if url not in pages:
            raise httpx.ConnectError("no page stubbed")
        return f"<html><head><title>{pages[url]}</title></head><body></body></html>"

    monkeypatch.setattr(ats_module, "_get_json", fake_get_json)
    monkeypatch.setattr(ats_module, "fetch_page", fake_title_page)
    monkeypatch.setattr(pipeline_module, "fetch_page", no_page_fetch)
    return responses, pages, calls


# --- match_ats ----------------------------------------------------------------

def test_match_ats_ids():
    m = match_ats("https://job-boards.greenhouse.io/greenhouse/jobs/8211828?gh_jid=8211828")
    assert (m.ats, m.board, m.job) == ("greenhouse", "greenhouse", "8211828")
    m = match_ats(f"https://jobs.lever.co/acme/{UUID}")
    assert (m.ats, m.board, m.job) == ("lever", "acme", UUID)
    m = match_ats(f"https://jobs.ashbyhq.com/acme/{UUID}")
    assert (m.ats, m.board, m.job) == ("ashby", "acme", UUID)
    assert match_ats("https://jobs.lever.co/acme") is None
    assert match_ats("https://careers.acme.com/jobs/1") is None


# --- adapters -----------------------------------------------------------------

def test_greenhouse(api):
    responses, _, calls = api
    responses["boards-api.greenhouse.io"] = GREENHOUSE_JSON
    url = "https://job-boards.greenhouse.io/greenhouse/jobs/8211828?gh_jid=8211828"
    job = build_posting(url)
    assert calls == ["https://boards-api.greenhouse.io/v1/boards/greenhouse/jobs/8211828?pay_transparency=true"]
    assert job.id == job_id_from_url(url)
    assert job.classified_by == "url_rule" and job.source == "pasted_url"
    assert (job.title, job.company, job.location) == ("Senior Software Engineer", "Greenhouse", "Ontario")
    assert job.work_mode is None
    assert job.salary_text == "CAD 129,000–164,500"
    assert str(job.date_posted) == "2026-09-10"
    assert job.description == "Build hiring tools.\n\n- Ruby\n- React"


def test_lever(api):
    responses, pages, calls = api
    responses["api.lever.co"] = LEVER_JSON
    url = f"https://jobs.lever.co/acme/{UUID}"
    pages[url] = "Acme Robotics - Backend Engineer"
    job = build_posting(url)
    assert calls == [f"https://api.lever.co/v0/postings/acme/{UUID}", url]
    assert job.title == "Backend Engineer" and job.company == "Acme Robotics"
    assert job.location == "San Francisco, CA"
    assert job.work_mode == "hybrid"
    assert job.salary_text == "USD 150,000–190,000 / per-year-salary"
    assert job.date_posted is not None
    assert job.description == "We build things.\n\nRequirements\n\n- Python\n- Postgres\n\nBenefits included."


def test_lever_eu_host(api):
    responses, _, calls = api
    responses["api.eu.lever.co"] = LEVER_JSON
    build_posting(f"https://jobs.eu.lever.co/acme/{UUID}")
    assert calls[0].startswith("https://api.eu.lever.co/")


def test_ashby(api):
    responses, pages, _ = api
    responses["api.ashbyhq.com"] = ASHBY_JSON
    url = f"https://jobs.ashbyhq.com/acme/{UUID}"
    pages[url] = "Staff Engineer @ Acme &amp; Co"
    job = build_posting(url)
    assert job.title == "Staff Engineer"
    assert job.company == "Acme & Co"
    assert job.work_mode == "remote"
    assert job.salary_text == "$180K - $220K"
    assert str(job.date_posted) == "2026-09-01"
    assert job.description == "Agents and data."


def test_ashby_job_not_on_board(api):
    responses, _, _ = api
    responses["api.ashbyhq.com"] = {"jobs": []}
    with pytest.raises(ExtractionError, match="not found"):
        build_posting(f"https://jobs.ashbyhq.com/acme/{UUID}")


def test_empty_description_fails(api):
    responses, _, _ = api
    responses["boards-api.greenhouse.io"] = {**GREENHOUSE_JSON, "content": ""}
    with pytest.raises(ExtractionError, match="no description"):
        build_posting("https://boards.greenhouse.io/greenhouse/jobs/1")


# --- non-ATS route ------------------------------------------------------------

def test_non_ats_url_uses_page_json_ld(monkeypatch):
    page = ('<script type="application/ld+json">'
            '{"@type": "JobPosting", "title": "Engineer", "description": "<p>Hi</p>"}</script>')
    monkeypatch.setattr(pipeline_module, "fetch_page", lambda url: page)
    job = build_posting("https://careers.acme.com/jobs/1")
    assert job.classified_by == "json_ld" and job.title == "Engineer"


def test_non_ats_url_without_posting_fails(monkeypatch):
    monkeypatch.setattr(pipeline_module, "fetch_page", lambda url: "<html><body>Jobs list</body></html>")
    with pytest.raises(ExtractionError, match="not recognized"):
        build_posting("https://careers.acme.com/jobs")


def test_company_page_failure_does_not_block(api):
    responses, _, _ = api  # no page stubbed -> fetch fails
    responses["api.ashbyhq.com"] = ASHBY_JSON
    job = build_posting(f"https://jobs.ashbyhq.com/acme/{UUID}")
    assert job.company is None and job.title == "Staff Engineer"


def test_application_suffix_trimmed(api):
    responses, pages, _ = api
    responses["api.ashbyhq.com"] = ASHBY_JSON
    job = build_posting(f"https://jobs.ashbyhq.com/acme/{UUID}/application")
    assert job.url == f"https://jobs.ashbyhq.com/acme/{UUID}"


@pytest.mark.parametrize(
    "ats, page_title, job_title, expected",
    [
        ("ashby", "Senior Backend Engineer - Backend Platform (USA Only) @ Close",
         "Senior Backend Engineer - Backend Platform (USA Only)", "Close"),
        ("ashby", "Engineer @ Growth @ Acme", "Engineer @ Growth", "Acme"),  # '@' inside the job title
        ("ashby", "Jobs", "Engineer", None),
        ("lever", "Acme - Senior Engineer - Platform", "Senior Engineer - Platform", "Acme"),
        ("lever", "Senior Engineer", "Senior Engineer", None),
    ],
)
def test_company_from_page_title(ats, page_title, job_title, expected):
    assert company_from_page_title(ats, page_title, job_title) == expected
