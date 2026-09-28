import json
from pathlib import Path

import pytest

from lodestar.fit_agent.parse import FitParseError, parse_fit_analysis, unknown_evidence
from lodestar.schemas import load_profile

EXAMPLE_PROFILE = Path(__file__).resolve().parents[1] / "profile.example.yaml"

GOOD = {
    "job_id": "job_1",
    "requirements": [
        {"requirement": "Python", "priority": "must_have", "evidence": ["proj-quizbot"],
         "match_level": "direct", "explanation": "Built QuizBot in Python."},
        {"requirement": "Go", "priority": "nice_to_have", "evidence": ["exp-acme", "exp-made-up"],
         "match_level": "related", "explanation": "Java backend."},
    ],
    "summary": "Good fit.",
}


def test_plain_json():
    assert len(parse_fit_analysis(json.dumps(GOOD), "job_1").requirements) == 2


def test_fenced_json_with_prose():
    text = "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nDone."
    assert parse_fit_analysis(text, "job_1").summary == "Good fit."


def test_missing_job_id_filled_in():
    data = {k: v for k, v in GOOD.items() if k != "job_id"}
    assert parse_fit_analysis(json.dumps(data), "job_1").job_id == "job_1"


@pytest.mark.parametrize(
    "text, message",
    [
        ("no json here", "no JSON object"),
        ("{not: valid}", "invalid JSON"),
        (json.dumps({**GOOD, "job_id": "job_2"}), "doesn't match the requested"),
        (json.dumps({**GOOD, "requirements": [{**GOOD["requirements"][0], "match_level": "partial"}]}), "schema"),
        (json.dumps({**GOOD, "requirements": []}), "no requirements"),
    ],
)
def test_bad_replies(text, message):
    with pytest.raises(FitParseError, match=message):
        parse_fit_analysis(text, "job_1")


def test_score_field_is_dropped():
    # FitAnalysis has no score field, so a score the agent adds anyway never reaches storage
    data = {**GOOD, "overall_score": 0.9}
    analysis = parse_fit_analysis(json.dumps(data), "job_1")
    assert not hasattr(analysis, "overall_score")


def test_unknown_evidence():
    analysis = parse_fit_analysis(json.dumps(GOOD), "job_1")
    assert unknown_evidence(analysis, load_profile(EXAMPLE_PROFILE)) == [("Go", "exp-made-up")]
