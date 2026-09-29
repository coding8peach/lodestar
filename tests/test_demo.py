"""Demo mode (read-only UI) and the demo database build, with fake agents."""

import importlib.util
from contextlib import closing
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lodestar.app import service
from lodestar.schemas import FitAgentReply, FitAnalysis, RequirementMatch
from lodestar.schemas.resume import ResumeAgentReply, TailoredResume

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_demo", ROOT / "demo" / "build_demo.py")
build_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_demo)


def _req(text, level, hard=False):
    return RequirementMatch(requirement=text, priority="must_have", evidence=[], match_level=level,
                            explanation="", hard_constraint=hard)


async def fake_analyze(job_id, max_paid_usd=None):
    from lodestar.db import connect, get_job
    with closing(connect()) as conn:
        title, company = get_job(conn, job_id).title, get_job(conn, job_id).company
    if company == "Driftwood Analytics":
        reqs = [_req("Python", "direct"), _req("Right to work in the UK", "gap", hard=True)]
    elif "Web Client" in title:
        reqs = [_req("React", "related"), _req("WebGL", "gap"), _req("CSS", "gap")]
    elif company in ("Harborline", "Tessellate AI"):
        reqs = [_req("Python", "direct"), _req("APIs", "direct")]
    else:
        reqs = [_req("Backend", "direct"), _req("Stack", "related")]
    return FitAgentReply(analysis=FitAnalysis(job_id=job_id, requirements=reqs, summary="s"), model="fake")


async def fake_tailor(job_id, max_paid_usd=None):
    resume = TailoredResume.model_validate({
        "job_id": job_id, "headline": "Backend engineer: Java and Python",
        "summary": {"text": "Backend engineer building Java services and Python agent systems.",
                    "sources": ["exp-latticeworks", "proj-tracklight"]},
        "skills": ["Python", "Java", "REST APIs"],
        "experience": [{"entry_id": "exp-latticeworks", "highlights": [
            {"text": "Designed and built REST APIs in Java for the experiment-data platform",
             "sources": ["exp-latticeworks"]}]}],
        "projects": [{"entry_id": "proj-tracklight", "highlights": [
            {"text": "Built a job-search assistant with LangGraph and an ADK agent", "sources": ["proj-tracklight"]}]}],
    })
    return ResumeAgentReply(resume=resume, model="fake")


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.delenv("LODESTAR_DEMO", raising=False)
    db = tmp_path / "demo.sqlite"
    result = build_demo.build(fake_analyze, fake_tailor, db=db)
    return db, result


def test_build_demo(built):
    db, result = built
    assert result["queued"] == 12 and result["skipped"] == 4       # iOS, solutions, manager, London
    assert result["analyzed"] == build_demo.ANALYZE and result["failed"] == 0
    assert result["approved"] >= 2 and result["tailored"] == build_demo.TAILOR
    from lodestar.db import connect
    with closing(connect(db)) as conn:
        statuses = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
        blocked = conn.execute("SELECT d.note FROM decisions d JOIN fit_results f ON f.id = d.fit_result_id "
                               "JOIN jobs j ON j.id = f.job_id WHERE j.company = 'Driftwood Analytics'").fetchone()
    assert statuses["resume_tailored"] == 2 and statuses["queued"] >= 1
    assert blocked[0].startswith("blocked: Right to work in the UK")
    assert not Path(f"{db}-wal").exists() or Path(f"{db}-wal").stat().st_size == 0  # one self-contained file


# --- demo mode ------------------------------------------------------------------------------

@pytest.fixture
def demo(built, monkeypatch):
    db, _ = built
    monkeypatch.setenv("LODESTAR_DEMO", "1")
    monkeypatch.setenv("LODESTAR_DB", str(db))
    monkeypatch.setenv("LODESTAR_PROFILE", str(ROOT / "demo" / "profile.yaml"))
    return db


def run_page(name):
    def script(page_name):
        from lodestar.ui import pages
        getattr(pages, page_name)()
    at = AppTest.from_function(script, args=(name,), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_review_is_read_only(demo):
    at = run_page("review_page")
    for key in ("approve", "reject", "skip"):
        assert at.button(key=key).disabled


def test_queue_is_read_only(demo):
    at = run_page("queue_page")
    assert at.button(key="discover").disabled and at.button(key="analyze").disabled
    assert any("Demo" in c.value for c in at.caption)


def test_resume_panel_is_read_only_but_shows_the_resume(demo):
    def script():
        from lodestar.app import service as svc
        from lodestar.ui import pages
        pages._resume_panel(next(i for i in svc.decisions("approve") if i.has_resume))
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception and at.button(key="tailor").disabled
    assert any("Morgan Lee" in m.value for m in at.markdown)


def test_service_refuses_writes_in_demo_mode(demo):
    job_id = service.review_queue()[0].job_id if service.review_queue() else "job_x"
    with pytest.raises(service.DemoReadOnly):
        service.record_decision(job_id, "approve")
    with pytest.raises(service.DemoReadOnly):
        service.analyze_next(1)
    with pytest.raises(service.DemoReadOnly):
        service.discover()
    with pytest.raises(service.DemoReadOnly):
        service.dismiss(["job_x"], "no")


def test_app_shows_the_demo_banner(demo):
    at = AppTest.from_file(str(ROOT / "src" / "lodestar" / "ui" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert "fictional candidate" in at.info[0].value


def test_committed_demo_files_are_valid():
    import yaml

    from lodestar.schemas import load_profile
    assert load_profile(ROOT / "demo" / "profile.yaml").name == "Morgan Lee"
    assert len(yaml.safe_load((ROOT / "demo" / "postings.yaml").read_text())) == 16
