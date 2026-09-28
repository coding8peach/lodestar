"""Shared fixtures: a seeded database for the service-layer and UI tests."""

import shutil
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from lodestar.db import connect, init_db, save_fit_result, save_job
from lodestar.schemas import FitAnalysis, JobPosting, RequirementMatch, job_id_from_url
from lodestar.scoring import score_analysis

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"
BODY = "Python backend services on PostgreSQL. " * 20


def add_job(n: int, title: str, company: str = "Acme", levels=None, hard_gap=False, posted=date(2026, 9, 1)) -> str:
    url = f"https://boards.greenhouse.io/acme/jobs/{n}"
    jid = job_id_from_url(url)
    with closing(connect()) as conn:
        init_db(conn)
        save_job(conn, JobPosting(id=jid, url=url, source="greenhouse", classified_by="url_rule", title=title,
                                  company=company, location="San Francisco", description=BODY, date_posted=posted,
                                  fetched_at=datetime(2026, 9, 27, tzinfo=timezone.utc)))
        if levels is not None:
            reqs = [RequirementMatch(requirement=f"r{i}", priority="must_have", evidence=[], match_level=lv,
                                     explanation="") for i, lv in enumerate(levels)]
            if hard_gap:
                reqs.append(RequirementMatch(requirement="Eligible to work in Canada", priority="must_have",
                                             evidence=[], match_level="gap", explanation="", hard_constraint=True))
            save_fit_result(conn, score_analysis(FitAnalysis(job_id=jid, requirements=reqs, summary="s")),
                            model="gemini-test")
    return jid


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("LODESTAR_DB", str(tmp_path / "t.sqlite"))
    profile = tmp_path / "profile.yaml"
    shutil.copy(EXAMPLE_PROFILE, profile)
    monkeypatch.setenv("LODESTAR_PROFILE", str(profile))
    return {
        "top": add_job(1, "Senior Backend Engineer", "Acme", ["direct", "direct"]),
        "possible": add_job(2, "Senior Platform Engineer", "Beta", ["related", "gap"]),
        "blocked": add_job(3, "Senior Engineer", "Gamma", ["direct"], hard_gap=True),
        "queued_senior": add_job(4, "Senior Software Engineer, Agents", "Acme"),
        "queued_staff": add_job(5, "Staff Software Engineer, Agents", "Acme"),
        "queued_other": add_job(6, "Backend Engineer", "Delta"),
    }
