from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from lodestar.schemas import JobPosting, job_id_from_url, normalize_url


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://jobs.lever.co/acme/abc-123/", "https://jobs.lever.co/acme/abc-123"),
        ("HTTPS://Jobs.Lever.co/acme/abc-123#apply", "https://jobs.lever.co/acme/abc-123"),
        (
            "https://boards.greenhouse.io/acme/jobs/42?gh_src=xyz&utm_source=linkedin",
            "https://boards.greenhouse.io/acme/jobs/42",
        ),
        # the job id lives in the query string here: it must survive
        ("https://acme.com/careers?gh_jid=42&utm_medium=x", "https://acme.com/careers?gh_jid=42"),
        # param order doesn't matter
        ("https://acme.com/job?b=2&a=1", "https://acme.com/job?a=1&b=2"),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_same_job_same_id():
    a = job_id_from_url("https://jobs.lever.co/acme/abc-123/?lever-source=x")
    b = job_id_from_url("https://jobs.lever.co/acme/abc-123")
    assert a == b and a.startswith("job_")


def test_different_query_job_ids_differ():
    assert job_id_from_url("https://acme.com/careers?gh_jid=1") != job_id_from_url(
        "https://acme.com/careers?gh_jid=2"
    )


def test_rejects_non_http():
    with pytest.raises(ValueError):
        normalize_url("ftp://example.com/job")


def make_posting(**overrides) -> dict:
    url = "https://jobs.lever.co/acme/abc-123"
    data = {
        "id": job_id_from_url(url),
        "url": url,
        "source": "pasted_url",
        "classified_by": "url_rule",
        "title": "Senior Backend Engineer",
        "description": "We are hiring...",
        "fetched_at": datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return data


def test_valid_posting():
    job = JobPosting.model_validate(make_posting(url="https://jobs.lever.co/acme/abc-123/?utm_source=x"))
    assert job.url == "https://jobs.lever.co/acme/abc-123"
    assert job.company is None


def test_id_url_mismatch_fails():
    with pytest.raises(ValidationError, match="does not match url"):
        JobPosting.model_validate(make_posting(id="job_0000000000000000"))


def test_naive_fetched_at_fails():
    with pytest.raises(ValidationError):
        JobPosting.model_validate(make_posting(fetched_at=datetime(2026, 9, 27, 12, 0)))


def test_fetched_at_converted_to_utc():
    pacific = timezone(timedelta(hours=-7))
    job = JobPosting.model_validate(make_posting(fetched_at=datetime(2026, 9, 27, 5, 0, tzinfo=pacific)))
    assert job.fetched_at.utcoffset() == timedelta(0) and job.fetched_at.hour == 12


def test_bad_classification_layer_fails():
    with pytest.raises(ValidationError):
        JobPosting.model_validate(make_posting(classified_by="regex"))


def test_requirements_field_not_allowed():
    with pytest.raises(ValidationError):
        JobPosting.model_validate(make_posting(requirements=["Python"]))
