"""ResumeService: the resume agent as served over A2A (same shape as the fit agent's FitService).

For each message: find the job_id; check the job and its fit analysis exist and read the
profile (MCP); run tailor_job with the per-run fallback and paid-budget filter; reply with
one ResumeAgentReply JSON message. Failures become failed A2A tasks.
"""

import logging
from collections.abc import AsyncGenerator, Callable

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.genai import types
from pydantic import Field

from lodestar.fit_agent.models import make_llm, validate_model_names
from lodestar.fit_agent.service import JOB_ID, _max_paid_usd, allowed_models
from lodestar.mcp_server.client import ToolFailed, call_tool
from lodestar.resume_agent.runner import tailor_job
from lodestar.schemas.profile import Profile
from lodestar.schemas.resume import ResumeAgentReply

log = logging.getLogger(__name__)

DESCRIPTION = (
    "Drafts a resume tailored to one saved, analyzed job, using only the candidate's profile. "
    "Send a job_id as the message text. Replies with JSON: {\"resume\": TailoredResume (every line "
    "cites profile entries; facts are filled in by the caller), \"model\": str, \"skipped\": [str], "
    "\"calls\": [LlmCall]}."
)


class ResumeService(BaseAgent):
    model_names: list[str] = Field(min_length=1)
    make_model: Callable[[str], BaseLlm] = make_llm

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        text = "".join(p.text or "" for p in (ctx.user_content.parts if ctx.user_content else []) or [])
        match = JOB_ID.search(text)
        if not match:
            raise ValueError(f"no job_id in the message: {text[:80]!r}")
        job_id = match.group(0)
        try:
            await call_tool("get_job", {"job_id": job_id})
            await call_tool("get_fit_analysis", {"job_id": job_id})
            profile = Profile.model_validate(await call_tool("get_profile", {}))
        except ToolFailed as e:
            raise ValueError(str(e)) from e

        models = allowed_models(validate_model_names(self.model_names), _max_paid_usd(ctx))
        run = await tailor_job(job_id, models, profile, make_model=self.make_model)
        reply = ResumeAgentReply(resume=run.result, model=run.model, skipped=run.skipped, tokens=run.tokens,
                                 calls=run.calls)
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=reply.model_dump_json())]))


def build_resume_service(model_names: list[str], make_model: Callable[[str], BaseLlm] = make_llm) -> ResumeService:
    return ResumeService(name="resume_agent", description=DESCRIPTION, model_names=model_names,
                         make_model=make_model)
