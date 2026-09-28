"""Render each UI page against a seeded database with Streamlit's AppTest (no browser)."""

import pytest
from streamlit.testing.v1 import AppTest

from lodestar.app import service


def status_labels(at: AppTest) -> list[str]:
    """Labels of st.status boxes anywhere on the page."""
    found, stack = [], [at.main]
    while stack:
        node = stack.pop()
        if type(node).__name__ == "Status":
            found.append(node.label)
        children = getattr(node, "children", None)
        stack.extend(children.values() if isinstance(children, dict) else children or [])
    return found


def run_page(name: str) -> AppTest:
    def script(page_name):
        from lodestar.ui import pages
        getattr(pages, page_name)()

    at = AppTest.from_function(script, args=(name,), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", ["review_page", "queue_page", "decisions_page", "usage_page"])
def test_pages_render(seeded, page):
    run_page(page)


def test_review_shows_best_first_and_approve_saves(seeded):
    at = run_page("review_page")
    assert at.subheader[0].value == "Senior Backend Engineer"   # the top pick is selected first
    at.text_input(key=f"note_{seeded['top']}").input("apply this week")
    at.button(key="approve").click().run()
    assert not at.exception
    d = service.job_detail(seeded["top"])
    assert d.status == "approved" and d.decision.note == "apply this week"
    assert at.subheader[0].value == "Senior Platform Engineer"  # moved on to the next job


def test_review_reject_blocked_needs_confirmation(seeded):
    at = run_page("review_page")
    assert at.button(key="reject_blocked").disabled
    at.checkbox(key="confirm_blocked").check().run()
    at.button(key="reject_blocked").click().run()
    assert service.job_detail(seeded["blocked"]).status == "rejected"


def test_empty_review_page_points_to_analyze(tmp_path, monkeypatch):
    monkeypatch.setenv("LODESTAR_DB", str(tmp_path / "empty.sqlite"))
    at = run_page("review_page")
    assert "lodestar analyze" in at.info[0].value


# --- Phase 2: the Queue page's run section -------------------------------------------------

def test_analyze_disabled_when_the_agent_is_down(seeded, monkeypatch):
    from lodestar.fit_agent import client

    monkeypatch.setattr(client, "agent_status", lambda: (False, "fit agent not running at http://localhost:8001"))
    at = run_page("queue_page")
    assert at.button(key="analyze").disabled
    assert any("not running" in c.value for c in at.caption)


def test_analyze_button_runs_a_batch(seeded, monkeypatch):
    from lodestar.app.models import AnalyzedJob, BatchResult
    from lodestar.fit_agent import client

    monkeypatch.setattr(client, "agent_status", lambda: (True, "fit agent running"))
    calls = []

    def fake_analyze_next(budget, on_progress=None, analyze_fn=None):
        calls.append(budget)
        item = AnalyzedJob(job_id="job_x", company="Acme", title="Senior Engineer", recommendation="top_pick",
                           score=0.8)
        on_progress(item)
        return BatchResult(run_id=1, results=[item])

    monkeypatch.setattr(service, "analyze_next", fake_analyze_next)
    at = run_page("queue_page")
    at.button(key="analyze").click().run()
    assert not at.exception and calls == [3]
    assert any("Ready on the Review page" in label for label in status_labels(at))


def test_find_new_postings_button(seeded, monkeypatch):
    from lodestar.app.models import BoardSummary, DiscoverySummary

    monkeypatch.setattr(service, "discover", lambda: DiscoverySummary(
        boards=[BoardSummary(company="Acme", listed=5, seen=3, queued=1, skipped={"title: android": 1})],
        queued_jobs=[("Acme", "Senior Engineer", "SF")], dry_run=False))
    at = run_page("queue_page")
    at.button(key="discover").click().run()
    assert not at.exception
    assert "Found 2 new posting(s): queued 1, filtered out 1" in status_labels(at)


def test_queue_table_has_arrow_friendly_types(seeded):
    # a column mixing ints and "" made Streamlit log an Arrow conversion error on every render
    import pandas as pd
    import pyarrow as pa

    items = service.ranked_queue()
    pa.Table.from_pandas(pd.DataFrame({"Similar": [q.similar_count or None for q in items]}))  # must not raise


def test_dollar_amounts_are_not_read_as_math(seeded, monkeypatch):
    from lodestar.fit_agent import client

    monkeypatch.setattr(client, "agent_status", lambda: (True, "fit agent running"))
    at = run_page("queue_page")
    assert any("\\$0.00 spent of \\$0.00 allowed" in c.value for c in at.caption)
