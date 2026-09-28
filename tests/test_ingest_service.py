from datetime import date, datetime, timedelta, timezone

import pytest

from lodestar.db import connect, get_job, get_status, init_db
from lodestar.ingest import ExtractionError, ingest_url, strip_apply_suffix, validate_job
from lodestar.ingest import service as service_module
from lodestar.schemas import JobPosting, job_id_from_url

UUID = "3f2a9c1e-1234-4abc-9def-0123456789ab"
URL = f"https://jobs.ashbyhq.com/acme/{UUID}"
LONG_TEXT = "We build agent tooling in Python on PostgreSQL. " * 10


def make_job(url: str = URL, **overrides) -> JobPosting:
    data = dict(
        id=job_id_from_url(url), url=url, source="pasted_url", classified_by="url_rule",
        title="Senior Backend Engineer ", company="  Acme ", location="Remote\n - US",
        salary_text=None, description=f"\n  {LONG_TEXT}  \n",
        fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc),
    )
    data.update(overrides)
    return JobPosting(**data)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def fetches(monkeypatch):
    """Stub build_posting; record every URL it's asked to fetch."""
    calls: list[str] = []

    def fake_build(url, source="pasted_url"):
        calls.append(url)
        return make_job(url)

    monkeypatch.setattr(service_module, "build_posting", fake_build)
    return calls


# --- validate_job ---------------------------------------------------------------

def test_validate_trims_whitespace():
    job = validate_job(make_job())
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme"
    assert job.location == "Remote - US"
    assert job.description == LONG_TEXT.strip()


def test_validate_rejects_short_description():
    with pytest.raises(ExtractionError, match="doesn't look like a full posting"):
        validate_job(make_job(description="Apply now!"))


def test_validate_warns_on_expired_posting(caplog):
    validate_job(make_job(valid_through=date.today() - timedelta(days=1)))
    assert "may be closed" in caplog.text


def test_blank_company_becomes_none():
    assert validate_job(make_job(company="   ")).company is None


# --- strip_apply_suffix -----------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (f"{URL}/application", URL),
        (f"{URL}/application/", URL),
        (f"https://jobs.lever.co/acme/{UUID}/apply", f"https://jobs.lever.co/acme/{UUID}"),
        ("https://careers.acme.com/jobs/1/apply", "https://careers.acme.com/jobs/1/apply"),  # not an ATS host
        (URL, URL),
    ],
)
def test_strip_apply_suffix(raw, expected):
    assert strip_apply_suffix(raw) == expected


# --- ingest_url -----------------------------------------------------------------

def test_ingest_saves_new_job(conn, fetches):
    result = ingest_url(conn, URL)
    assert result.status == "saved"
    assert result.job_id == job_id_from_url(URL)
    assert (result.title, result.company) == ("Senior Backend Engineer", "Acme")
    assert get_job(conn, result.job_id).title == "Senior Backend Engineer"  # cleaned before saving
    assert get_status(conn, result.job_id) == "queued"


def test_ingest_again_does_not_refetch(conn, fetches):
    ingest_url(conn, URL)
    again = ingest_url(conn, f"{URL}/application?utm_source=x")  # same posting, different spelling
    assert again.status == "already_saved"
    assert fetches == [URL]


def test_ingest_bad_url(conn, fetches):
    with pytest.raises(ValueError):
        ingest_url(conn, "not a url")
    assert fetches == []


def test_ingest_extraction_failure_saves_nothing(conn, monkeypatch):
    def failing(url, source="pasted_url"):
        raise ExtractionError("no posting here")

    monkeypatch.setattr(service_module, "build_posting", failing)
    with pytest.raises(ExtractionError):
        ingest_url(conn, URL)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
