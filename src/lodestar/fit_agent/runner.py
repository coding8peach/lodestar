"""Run a fit analysis for one job, falling back to the next model per run.

Each attempt is a complete analysis on one model: a fresh agent, a fresh session,
a fresh MCP connection. If that model hits a rate limit, a size limit or an outage
at any turn, the next model starts over from the beginning. Providers leave their
own artifacts in a conversation (reasoning fields, thought signatures), so a
history started by one can't safely be continued by another.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from lodestar.fit_agent.agent import build_fit_agent
from lodestar.fit_agent.models import is_fallback_error, is_overloaded, make_llm, validate_model_names
from lodestar.fit_agent.parse import FitParseError, parse_fit_analysis
from lodestar.fit_agent.usage import UsageRecorder
from lodestar.schemas.fit import FitAnalysis, LlmCall

log = logging.getLogger(__name__)

APP_NAME = "lodestar_fit"
USER_ID = "candidate"


def describe_error(e: BaseException) -> str:
    """One line for a provider error. Some wrappers have an empty message, so walk the
    cause chain for the first one that says something, and include any status code."""
    seen: set[int] = set()
    current: BaseException | None = e
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = (str(current).splitlines() or [""])[0].strip()
        if text:
            break
        current = current.__cause__ or current.__context__
    else:
        text = ""
    code = next((getattr(x, a) for x in (e, e.__cause__, e.__context__) if x is not None
                 for a in ("status_code", "code") if isinstance(getattr(x, a, None), int)), None)
    parts = [type(e).__name__]
    if code:
        parts.append(f"HTTP {code}")
    if "RESOURCE_EXHAUSTED" in text or "ResourceExhausted" in type(e).__name__:
        parts.append("quota or rate limit reached")
    detail = f": {text[:200]}" if text else ""
    return " ".join(parts) + detail


# Waits before retrying the same model after a temporary overload (503 "high demand").
# Rate limits and quotas don't retry; they fall back to the next model right away.
OVERLOAD_RETRY_DELAYS = (5.0, 15.0)


class AllModelsFailed(RuntimeError):
    """Every model in the list failed with a rate-limit, size or outage error."""

    def __init__(self, message: str, calls: list[LlmCall] | None = None):
        super().__init__(message)
        self.calls = calls or []


@dataclass
class FitRun:
    analysis: FitAnalysis
    model: str                      # the one model that produced this analysis
    tokens: int                     # total tokens reported for the successful attempt
    skipped: list[str] = field(default_factory=list)  # "model: reason" for models that fell through
    calls: list[LlmCall] = field(default_factory=list)  # every model call, all attempts


async def _ask(runner: InMemoryRunner, session_id: str, text: str) -> tuple[str, int]:
    """Send one message; return (final text, total tokens)."""
    message = types.Content(role="user", parts=[types.Part(text=text)])
    final, tokens = "", 0
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        if event.usage_metadata and event.usage_metadata.total_token_count:
            tokens += event.usage_metadata.total_token_count
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts if not p.thought)
    return final, tokens


async def _analyze_with(model: BaseLlm, job_id: str, usage: UsageRecorder | None = None) -> tuple[FitAnalysis, int]:
    """One complete attempt on one model. A malformed reply gets one retry on the same model."""
    runner = InMemoryRunner(agent=build_fit_agent(model, usage), app_name=APP_NAME)
    try:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        text, tokens = await _ask(runner, session.id, f"Analyze the fit for job_id {job_id}.")
        try:
            return parse_fit_analysis(text, job_id), tokens
        except FitParseError as e:
            log.warning("reply wasn't a valid FitAnalysis (%s); asking once more", e)
            retry = f"Your reply was not valid: {e}. Reply again with only the JSON object in the required shape."
            text, more = await _ask(runner, session.id, retry)
            return parse_fit_analysis(text, job_id), tokens + more  # a second failure is raised
    finally:
        await runner.close()


async def analyze_job(
    job_id: str, model_names: list[str], make_model: Callable[[str], BaseLlm] = make_llm
) -> FitRun:
    """Try each model in order until one completes the analysis."""
    skipped: list[str] = []
    calls: list[LlmCall] = []
    for attempt, name in enumerate(validate_model_names(model_names), start=1):
        native = name.removeprefix("gemini/").startswith("gemini")
        waits = list(OVERLOAD_RETRY_DELAYS)
        while True:
            log.info("analyzing %s with %s", job_id, name)
            usage = UsageRecorder(name, attempt, native_gemini=native)
            try:
                analysis, tokens = await _analyze_with(make_model(name), job_id, usage)
            except Exception as e:
                calls += usage.calls
                if not is_fallback_error(e):
                    raise
                reason = describe_error(e)
                if waits and is_overloaded(e):
                    delay = waits.pop(0)
                    log.warning("%s busy (%s); retrying in %.0fs", name, reason, delay)
                    await asyncio.sleep(delay)
                    continue  # same model, fresh attempt
                log.warning("%s unavailable (%s); starting over with the next model", name, reason)
                skipped.append(f"{name}: {reason}")
                break
            calls += usage.calls
            return FitRun(analysis=analysis, model=name, tokens=tokens, skipped=skipped, calls=calls)
    raise AllModelsFailed("every model failed: " + " | ".join(skipped), calls=calls)
