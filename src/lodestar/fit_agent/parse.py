"""Turn the agent's final text into a FitAnalysis, and check its evidence against the profile."""

import json

from lodestar.schemas.fit import FitAnalysis
from lodestar.schemas.profile import Profile


class FitParseError(ValueError):
    """The agent's reply isn't a valid FitAnalysis."""


def _json_object(text: str) -> str:
    """The outermost {...} in the reply, ignoring markdown fences or stray prose around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise FitParseError("no JSON object in the reply")
    return text[start : end + 1]


def parse_fit_analysis(text: str, job_id: str) -> FitAnalysis:
    try:
        data = json.loads(_json_object(text))
    except json.JSONDecodeError as e:
        raise FitParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise FitParseError("the JSON is not an object")
    data.setdefault("job_id", job_id)
    if data["job_id"] != job_id:
        raise FitParseError(f"job_id {data['job_id']!r} doesn't match the requested {job_id!r}")
    try:
        analysis = FitAnalysis.model_validate(data)
    except ValueError as e:
        raise FitParseError(f"doesn't match the FitAnalysis schema: {e}") from e
    if not analysis.requirements:
        raise FitParseError("no requirements listed")
    return analysis


def unknown_evidence(analysis: FitAnalysis, profile: Profile) -> list[tuple[str, str]]:
    """(requirement, id) for every evidence id that isn't an entry in the profile."""
    known = {x.id for x in [*profile.experience, *profile.projects, *profile.education]}
    return [(m.requirement, ref) for m in analysis.requirements for ref in m.evidence if ref not in known]
