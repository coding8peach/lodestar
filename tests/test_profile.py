from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lodestar.schemas import Profile, load_profile

EXAMPLE = Path(__file__).resolve().parents[1] / "profile.example.yaml"


def example_data() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_example_profile_loads():
    profile = load_profile(EXAMPLE)
    assert profile.targets.frontend_preference == "light_frontend_ok"
    java = next(s for s in profile.skills if s.name == "Java")
    assert [e.id for e in profile.evidence_for(java)] == ["exp-acme"]


def test_unknown_evidence_id_fails():
    data = example_data()
    data["skills"][0]["evidence"] = ["exp-typo"]
    with pytest.raises(ValidationError, match="unknown evidence ids"):
        Profile.model_validate(data)


def test_unknown_key_fails():
    data = example_data()
    data["skills"][0]["lastused"] = 2020
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_duplicate_ids_fail():
    data = example_data()
    data["projects"][0]["id"] = "exp-acme"
    with pytest.raises(ValidationError, match="duplicate ids"):
        Profile.model_validate(data)


def test_duplicate_skill_fails():
    data = example_data()
    data["skills"].append({"name": "java", "depth": "course", "last_used": 2025, "evidence": []})
    with pytest.raises(ValidationError, match="more than once"):
        Profile.model_validate(data)


def test_bad_depth_fails():
    data = example_data()
    data["skills"][0]["depth"] = "expert"
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


def test_bad_start_date_fails():
    data = example_data()
    data["experience"][0]["start"] = "March 2012"
    with pytest.raises(ValidationError):
        Profile.model_validate(data)
