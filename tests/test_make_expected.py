import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "make_expected.py"
_spec = importlib.util.spec_from_file_location("make_expected", _PATH)
me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(me)


@pytest.fixture
def dataset(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for name, title in [("job_01", "Backend Engineer"), ("job_02", "Staff Engineer")]:
        (jobs / f"{name}.json").write_text(json.dumps({"title": title, "company": "Acme", "location": "Remote"}))
    return jobs, tmp_path / "expected.yaml"


def fill(expected, name, decision="apply", reason="Python backend"):
    """Replace one stub block with a filled-in entry, the way a person would edit it."""
    text = expected.read_text()
    start = text.index(f"{name}:  # ")
    end = text.find("\njob_", start)
    end = len(text) if end == -1 else end + 1
    entry = f'{name}:\n  human_decision: {decision}\n  reason: "{reason}"\n'
    expected.write_text(text[:start] + entry + text[end:])


def test_adds_stubs_for_all_jobs(dataset):
    jobs, expected = dataset
    assert me.add_stubs(jobs, expected) == ["job_01", "job_02"]
    text = expected.read_text()
    assert "job_01:  # Backend Engineer | Acme | Remote" in text
    assert set(me.load_expected(expected)) == {"job_01", "job_02"}


def test_second_run_adds_nothing_and_keeps_edits(dataset):
    jobs, expected = dataset
    me.add_stubs(jobs, expected)
    fill(expected, "job_01")
    expected.write_text(expected.read_text() + "# my own note\n")
    before = expected.read_text()
    assert me.add_stubs(jobs, expected) == []
    assert expected.read_text() == before


def test_new_job_appended_after_existing_entries(dataset):
    jobs, expected = dataset
    me.add_stubs(jobs, expected)
    fill(expected, "job_01")
    (jobs / "job_03.json").write_text(json.dumps({"title": "New"}))
    assert me.add_stubs(jobs, expected) == ["job_03"]
    data = me.load_expected(expected)
    assert data["job_01"]["human_decision"] == "apply"
    assert "job_03" in data


def test_check_flags_blank_stubs(dataset):
    jobs, expected = dataset
    me.add_stubs(jobs, expected)
    problems = me.check(jobs, expected)
    assert any("job_01: human_decision" in p for p in problems)
    assert any("job_02: reason is empty" in p for p in problems)


def test_check_passes_when_filled(dataset):
    jobs, expected = dataset
    me.add_stubs(jobs, expected)
    fill(expected, "job_01", "apply")
    fill(expected, "job_02", "skip", "Mostly Go")
    assert me.check(jobs, expected) == []


def test_check_flags_bad_decision_and_missing_files(dataset):
    jobs, expected = dataset
    me.add_stubs(jobs, expected)
    fill(expected, "job_01", "definitely")
    fill(expected, "job_02")
    (jobs / "job_02.json").unlink()
    (jobs / "job_03.json").write_text(json.dumps({"title": "Unlisted"}))
    problems = me.check(jobs, expected)
    assert any("job_01: human_decision" in p for p in problems)
    assert any("job_02: entry has no job file" in p for p in problems)
    assert any("job_03: job file has no entry" in p for p in problems)


def test_stub_falls_back_to_board_and_shows_url(dataset):
    jobs, expected = dataset
    url = "https://jobs.ashbyhq.com/close/5bf4629b-d2ee-4129-980f-e31b96185c9c"
    (jobs / "job_03.json").write_text(json.dumps(
        {"title": "Senior Backend Engineer", "company": None, "location": "USA - Remote", "url": url}))
    me.add_stubs(jobs, expected)
    text = expected.read_text()
    assert "job_03:  # Senior Backend Engineer | close (board) | USA - Remote\n" in text
    assert f"  # {url}\n" in text
    assert me.load_expected(expected)["job_03"] == {"human_decision": None, "reason": ""}
