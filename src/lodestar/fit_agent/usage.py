"""Record every model call the fit agent makes: tokens, latency, and failures.

Attached to the LlmAgent as before/after/on-error model callbacks. The service only
collects; callers (the workflow, eval runs) store the calls and compute cost.
"""

import logging
import time
from datetime import datetime, timezone

from lodestar.fit_agent.models import is_fallback_error
from lodestar.schemas.fit import LlmCall

log = logging.getLogger(__name__)


class UsageRecorder:
    def __init__(self, model: str, attempt: int, native_gemini: bool):
        self.model = model
        self.attempt = attempt
        # Gemini reports thinking separately from output; LiteLLM's output already includes it.
        self.native_gemini = native_gemini
        self.calls: list[LlmCall] = []
        self._started: tuple[float, datetime] | None = None

    def _elapsed(self) -> tuple[int | None, datetime]:
        if self._started is None:
            return None, datetime.now(timezone.utc)
        t0, started_at = self._started
        self._started = None
        return int((time.monotonic() - t0) * 1000), started_at

    def before_model(self, callback_context, llm_request):
        self._started = (time.monotonic(), datetime.now(timezone.utc))
        return None  # never replace the call

    def after_model(self, callback_context, llm_response):
        latency, started_at = self._elapsed()
        usage = llm_response.usage_metadata
        prompt = (usage.prompt_token_count or 0) if usage else 0
        output = (usage.candidates_token_count or 0) if usage else 0
        thinking = (usage.thoughts_token_count or 0) if usage else 0
        billed_output = output + thinking if self.native_gemini else output
        call = LlmCall(model=self.model, attempt=self.attempt, status="ok", input_tokens=prompt,
                       output_tokens=billed_output, thinking_tokens=thinking, latency_ms=latency,
                       started_at=started_at)
        self.calls.append(call)
        log.info("llm call: %s ok, %d in / %d out tokens, %s ms", self.model, prompt, billed_output, latency)
        return None  # never replace the response

    def on_model_error(self, callback_context, llm_request, error):
        from lodestar.fit_agent.runner import describe_error  # runner imports this module

        latency, started_at = self._elapsed()
        status = "rate_limited" if is_fallback_error(error) else "error"
        self.calls.append(LlmCall(model=self.model, attempt=self.attempt, status=status, latency_ms=latency,
                                  error=describe_error(error)[:500], started_at=started_at))
        log.info("llm call: %s %s", self.model, status)
        return None  # let the error propagate; the runner decides whether to fall back
