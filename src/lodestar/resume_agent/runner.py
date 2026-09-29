"""Tailor one resume: resume agent + the shared per-run fallback, parse, and Python's checks."""

import json
from collections.abc import Callable

from google.adk.models.base_llm import BaseLlm
from pydantic import ValidationError

from lodestar.fit_agent.models import make_llm
from lodestar.fit_agent.parse import _json_object
from lodestar.fit_agent.runner import AgentRun, _attempt, run_with_fallback
from lodestar.resume.validate import ResumeRejected, check_resume
from lodestar.resume_agent.agent import build_resume_agent
from lodestar.schemas.profile import Profile
from lodestar.schemas.resume import TailoredResume


def parse_resume(text: str, job_id: str, profile: Profile) -> TailoredResume:
    """Reply text -> checked TailoredResume. Raises ResumeRejected listing every problem."""
    try:
        data = json.loads(_json_object(text))
        data.setdefault("job_id", job_id)
        draft = TailoredResume.model_validate(data)
    except (ValueError, ValidationError) as e:  # FitParseError from _json_object is a ValueError
        raise ResumeRejected(f"not a valid resume JSON object: {str(e)[:300]}") from e
    return check_resume(draft, profile, job_id)


async def tailor_job(
    job_id: str, model_names: list[str], profile: Profile, make_model: Callable[[str], BaseLlm] = make_llm
) -> AgentRun:
    """AgentRun whose result is a checked TailoredResume (required roles added by Python)."""
    async def attempt(model, usage):
        return await _attempt(build_resume_agent(model, usage), f"Draft a tailored resume for job_id {job_id}.",
                              lambda text: parse_resume(text, job_id, profile), (ResumeRejected,))

    return await run_with_fallback(job_id, model_names, attempt, make_model)
