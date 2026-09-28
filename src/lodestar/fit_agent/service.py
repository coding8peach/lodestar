"""FitService: the agent that is served over A2A.

A thin ADK agent around the per-run fallback. For each A2A message it:
  1. reads a job_id from the message text and checks the job exists (MCP get_job)
  2. runs analyze_job(): a complete analysis per model, falling back to the next
     model on rate limits / size limits / outages (see runner.py)
  3. replies with one message: a FitAgentReply as JSON (analysis, model, skipped, tokens)

Only a job_id comes in and only a validated result goes out, never a half-finished
conversation. Any failure is raised, which the A2A executor turns into a failed task
carrying the message.
"""

import logging
import re
from collections.abc import AsyncGenerator, Callable

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.genai import types
from pydantic import Field

from lodestar.fit_agent.models import make_llm, validate_model_names
from lodestar.fit_agent.runner import analyze_job
from lodestar.mcp_server.client import ToolFailed, call_tool
from lodestar.pricing import price_for
from lodestar.schemas.fit import FitAgentReply

log = logging.getLogger(__name__)

JOB_ID = re.compile(r"\bjob_[0-9a-f]{16}\b")

DESCRIPTION = (
    "Judges how well the candidate fits one saved job posting, requirement by requirement. "
    "Send a job_id (e.g. job_0a4f74b6c1690b60) as the message text. Replies with JSON: "
    '{"analysis": FitAnalysis (requirements + summary, no score), "model": str, '
    '"skipped": [str], "tokens": int}.'
)


class FitService(BaseAgent):
    model_names: list[str] = Field(min_length=1)
    make_model: Callable[[str], BaseLlm] = make_llm  # injectable for tests

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        text = "".join(p.text or "" for p in (ctx.user_content.parts if ctx.user_content else []) or [])
        match = JOB_ID.search(text)
        if not match:
            raise ValueError(f"no job_id in the message (expected something like job_0a4f74b6c1690b60): {text[:80]!r}")
        job_id = match.group(0)
        try:
            await call_tool("get_job", {"job_id": job_id})
        except ToolFailed as e:
            raise ValueError(str(e)) from e

        models = allowed_models(validate_model_names(self.model_names), _max_paid_usd(ctx))
        run = await analyze_job(job_id, models, make_model=self.make_model)
        reply = FitAgentReply(analysis=run.analysis, model=run.model, skipped=run.skipped, tokens=run.tokens,
                              calls=run.calls)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text=reply.model_dump_json())]),
        )


def _max_paid_usd(ctx: InvocationContext) -> float | None:
    """The caller's paid-model budget for this request, sent as A2A request metadata."""
    metadata = (ctx.run_config.custom_metadata or {}) if ctx.run_config else {}
    value = (metadata.get("a2a_metadata") or {}).get("max_paid_usd")
    return float(value) if value is not None else None


def allowed_models(model_names: list[str], max_paid_usd: float | None) -> list[str]:
    """Drop paid models when the caller's paid budget is used up.

    A model counts as free only when config/prices.yaml marks it free_tier; a model with an
    unknown price is treated as paid, to be safe.
    """
    if max_paid_usd is None or max_paid_usd > 0:
        return model_names
    free = [m for m in model_names if (p := price_for(m)) is not None and p.free_tier]
    if not free:
        raise ValueError("the paid-model budget for today is used up and no free model is configured "
                         "(mark free models with free_tier in config/prices.yaml)")
    if len(free) < len(model_names):
        log.info("paid budget used up: models limited to %s", ", ".join(free))
    return free


def build_fit_service(model_names: list[str], make_model: Callable[[str], BaseLlm] = make_llm) -> FitService:
    return FitService(name="fit_agent", description=DESCRIPTION, model_names=model_names, make_model=make_model)
