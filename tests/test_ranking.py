from datetime import date, datetime, timezone

from lodestar.ranking import RankConfig, group_key, normalize_title, pick, profile_keywords, rank_queue, score_job
from lodestar.schemas import JobPosting, Profile, job_id_from_url

PROFILE = Profile.model_validate({
    "name": "K", "summary": "s",
    "targets": {"roles": ["Backend"], "seniority": ["senior"], "locations": ["SF"], "work_modes": ["remote"],
                "frontend_preference": "light_frontend_ok"},
    "experience": [{"id": "exp-a", "company": "A", "title": "Staff", "start": "2010"}],
    "skills": [
        {"name": "Python", "depth": "project", "last_used": 2026, "evidence": ["exp-a"]},
        {"name": "Java", "depth": "production", "last_used": 2019, "evidence": ["exp-a"]},
        {"name": "C/C++", "depth": "production", "last_used": 2009, "evidence": ["exp-a"]},
        {"name": "MCP", "depth": "project", "last_used": 2026, "evidence": ["exp-a"]},
        {"name": "React", "depth": "production", "last_used": 2019, "evidence": ["exp-a"]},
    ],
})
CONFIG = RankConfig(max_per_company_per_run=2, penalize=("react", "frontend"))
N = iter(range(10_000))


def job(title, company="Harvey", body="", posted=date(2026, 9, 1)) -> JobPosting:
    url = f"https://jobs.ashbyhq.com/x/00000000-0000-0000-0000-{next(N):012d}"
    return JobPosting(id=job_id_from_url(url), url=url, source="ashby", classified_by="url_rule", title=title,
                      company=company, description=body or "x" * 300, date_posted=posted,
                      fetched_at=datetime(2026, 9, 28, tzinfo=timezone.utc))


def test_normalize_title_strips_seniority():
    assert normalize_title("Senior Software Engineer, Backend") == "software engineer backend"
    assert normalize_title("Staff Software Engineer, Backend") == "software engineer backend"
    assert normalize_title("Staff/Sr. Staff Software Engineer, Product") == "software engineer product"
    assert normalize_title("Sr. Systems Engineer") == "systems engineer"


def test_profile_keywords_split_and_weight():
    kw = profile_keywords(PROFILE)
    assert kw == {"python": 2, "java": 3, "c++": 3, "mcp": 2, "react": 3}  # "c" dropped: too short


def test_score_counts_title_and_description_and_penalizes_frontend():
    backend = job("Senior Backend Engineer (Python)", body="Python services. Java too. " + "x" * 300)
    frontend = job("Senior Frontend Engineer", body="React, React, react and Python. " + "x" * 300)
    s1, m1 = score_job(backend, profile_keywords(PROFILE), CONFIG.penalize)
    s2, m2 = score_job(frontend, profile_keywords(PROFILE), CONFIG.penalize)
    assert m1 == ["python", "java"] and s1 == 2 * (3 + 1) + 3 * 1
    assert "-react" in m2 and "-frontend" in m2 and "react" not in m2  # a skill, but penalized only
    assert s2 == 2 * 1 - (3 * 1) - (6 + 0)  # python in body; react x3 in body; frontend in title


def test_near_duplicates_grouped_best_kept():
    senior = job("Senior Software Engineer, Backend", body="Python " * 5 + "x" * 300)
    staff = job("Staff Software Engineer, Backend", body="Python " + "x" * 300)
    other = job("Staff Software Engineer, Agents", body="MCP " + "x" * 300)
    ranked, already = rank_queue([staff, senior, other], [], PROFILE, CONFIG)
    assert [r.job.title for r in ranked] == ["Senior Software Engineer, Backend", "Staff Software Engineer, Agents"]
    assert [j.title for j in ranked[0].similar] == ["Staff Software Engineer, Backend"]
    assert already == []


def test_group_already_analyzed_is_skipped():
    done = job("Senior Software Engineer, Backend")
    queued = job("Staff Software Engineer, Backend")
    ranked, already = rank_queue([queued], [done], PROFILE, CONFIG)
    assert ranked == [] and already == [(queued, done)]
    assert group_key(done) == group_key(queued)


def test_newer_posting_wins_a_tie():
    old = job("Backend Engineer", company="A", posted=date(2026, 8, 1))
    new = job("Platform Engineer", company="B", posted=date(2026, 9, 20))
    ranked, _ = rank_queue([old, new], [], PROFILE, CONFIG)
    assert [r.job.company for r in ranked] == ["B", "A"]


def test_pick_respects_budget_and_company_cap():
    jobs = [job(f"Engineer {i}", company="Harvey", body="Python " * (10 - i) + "x" * 300) for i in range(4)]
    jobs.append(job("Engineer X", company="Close", body="x" * 300))
    ranked, _ = rank_queue(jobs, [], PROFILE, CONFIG)
    picked = pick(ranked, budget=3, config=CONFIG)
    assert [r.job.company for r in picked] == ["Harvey", "Harvey", "Close"]
    assert pick(ranked, budget=0, config=CONFIG) == []
