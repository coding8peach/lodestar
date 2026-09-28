import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "compare_runs.py"
_spec = importlib.util.spec_from_file_location("compare_runs", _PATH)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

EXPECTED = {"job_01": {"human_decision": "apply"}, "job_02": {"human_decision": "skip"}}


def save(runs_dir: Path, label: str, job: str, score: float, rec: str, levels: list[str], tokens=100):
    folder = runs_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    reqs = [{"requirement": f"r{i}", "priority": "must_have", "evidence": [], "match_level": lv,
             "explanation": ""} for i, lv in enumerate(levels)]
    (folder / f"{job}.json").write_text(json.dumps({
        "job": job, "model": "m", "tokens": tokens,
        "result": {"overall_score": score, "recommendation": rec,
                   "analysis": {"job_id": job, "requirements": reqs, "summary": ""}},
    }))


@pytest.fixture
def runs(tmp_path):
    save(tmp_path, "base", "job_01", 0.78, "strong_fit", ["direct", "related"])
    save(tmp_path, "base", "job_02", 0.40, "weak_fit", ["gap", "gap"])
    for i, (s1, r1) in enumerate([(0.60, "possible_fit"), (0.80, "strong_fit"), (0.70, "possible_fit")], 1):
        save(tmp_path, f"new-r{i}", "job_01", s1, r1, ["related", "related"])
        save(tmp_path, f"new-r{i}", "job_02", 0.30, "weak_fit", ["gap"])
    return tmp_path


def test_single_folder_group(runs):
    data = cr.compare(["base"], runs, EXPECTED)
    cell = data["rows"][0]["cells"]["base"]
    assert cell == {"mean": 0.78, "min": 0.78, "max": 0.78, "recommendations": ["strong_fit"]}
    assert data["summaries"]["base"]["agreement"] == {"exact": 2, "one step": 0, "opposite": 0}


def test_pattern_group_aggregates_runs(runs):
    data = cr.compare(["base", "new-r*"], runs, EXPECTED)
    cell = data["rows"][0]["cells"]["new-r*"]
    assert cell["mean"] == 0.7 and (cell["min"], cell["max"]) == (0.6, 0.8)
    assert cell["recommendations"] == ["possible_fit", "strong_fit", "possible_fit"]
    s = data["summaries"]["new-r*"]
    assert s["runs"] == ["new-r1", "new-r2", "new-r3"]
    assert s["agreement"] == {"exact": 4, "one step": 2, "opposite": 0}  # 1 + 3 exact (job_02 weak/skip x3)
    assert s["analyses"] == 6
    assert s["levels_per_analysis"] == {"direct": 0.0, "related": 1.0, "gap": 0.5}
    assert s["requirements_per_analysis"] == 1.5 and s["tokens_per_analysis"] == 100


def test_cell_format(runs):
    data = cr.compare(["new-r*"], runs, EXPECTED)
    assert cr._cell(data["rows"][0]["cells"]["new-r*"]) == "0.70 [0.60-0.80] psp"
    assert cr._cell(None) == "-"


def test_job_missing_from_a_group(runs):
    save(runs, "other", "job_01", 0.7, "possible_fit", ["direct"])
    data = cr.compare(["base", "other"], runs, EXPECTED)
    assert "other" not in data["rows"][1]["cells"]


def test_no_match(runs):
    with pytest.raises(FileNotFoundError, match="no run folder matches"):
        cr.compare(["nope*"], runs, EXPECTED)


def test_agreement_mapping():
    assert cr.agreement("strong_fit", "apply") == "exact"
    assert cr.agreement("possible_fit", "apply") == "one step"
    assert cr.agreement("weak_fit", "apply") == "opposite"
    assert cr.agreement("weak_fit", None) is None


def test_overlapping_groups_rejected(runs):
    with pytest.raises(ValueError, match="matches both"):
        cr.compare(["new-r*", "new-r1"], runs, EXPECTED)


def test_rescore_applies_current_scoring(runs):
    # saved says strong_fit 0.78 for direct+related (must_have each): current math gives (2+1.2)/4 = 0.8
    data = cr.compare(["base"], runs, EXPECTED, rescore=True)
    assert data["rows"][0]["cells"]["base"]["mean"] == 0.8
    assert data["summaries"]["base"]["hard_constraint_data"] is False


def test_rescore_caps_unmet_hard_constraint(tmp_path):
    folder = tmp_path / "v2c"
    folder.mkdir()
    reqs = [{"requirement": "Python", "priority": "must_have", "evidence": [], "match_level": "direct",
             "explanation": "", "hard_constraint": False}] * 4 + [
            {"requirement": "Eligible to work in Canada", "priority": "must_have", "evidence": [],
             "match_level": "gap", "explanation": "", "hard_constraint": True}]
    (folder / "job_01.json").write_text(json.dumps({"job": "job_01", "model": "m", "tokens": 1, "result": {
        "overall_score": 0.8, "recommendation": "strong_fit",
        "analysis": {"job_id": "job_01", "requirements": reqs, "summary": ""}}}))
    data = cr.compare(["v2c"], tmp_path, EXPECTED, rescore=True)
    cell = data["rows"][0]["cells"]["v2c"]
    assert cell["mean"] == 1.0 and cell["recommendations"] == ["blocked"]  # skill fit 1.0, blocked
    assert data["summaries"]["v2c"]["hard_constraint_data"] is True


def test_incomplete_runs_average_per_analysis(tmp_path):
    save(tmp_path, "g-r1", "job_01", 0.7, "possible_fit", ["direct", "gap"], tokens=100)
    save(tmp_path, "g-r1", "job_02", 0.7, "possible_fit", ["direct", "gap"], tokens=100)
    save(tmp_path, "g-r2", "job_01", 0.7, "possible_fit", ["direct", "gap"], tokens=100)  # job_02 missing
    s = cr.compare(["g-r*"], tmp_path, EXPECTED)["summaries"]["g-r*"]
    assert s["analyses"] == 3 and s["requirements_per_analysis"] == 2.0 and s["tokens_per_analysis"] == 100


def test_new_recommendations_map_to_decisions():
    assert cr.agreement("top_pick", "apply") == "exact"
    assert cr.agreement("possible", "apply") == "one step"
    assert cr.agreement("blocked", "skip") == "exact"
    assert cr.agreement("blocked", "apply") == "opposite"
