"""Model configuration for the fit agent.

LODESTAR_FIT_MODELS in .env is a comma-separated list, tried in order one whole
run at a time (see runner.py). Names starting with "gemini" use ADK's native
Gemini support (GOOGLE_API_KEY); anything else goes through LiteLLM
(GROQ_API_KEY, OPENAI_API_KEY, ...).
"""

import logging
import os
import re
from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

log = logging.getLogger(__name__)

MODELS_ENV = "LODESTAR_FIT_MODELS"
_VALID_NAME = re.compile(r"^[A-Za-z0-9][\w.:/-]*$")

_FALLBACK_STATUS = {408, 413, 429, 500, 502, 503, 504}
_FALLBACK_NAME_HINTS = ("RateLimit", "ServiceUnavailable", "APIConnection", "Timeout", "InternalServer")


def validate_model_names(names: list[str]) -> list[str]:
    """Reject empty lists and placeholder-looking names before any API call."""
    if not names:
        raise ValueError(f"no models configured; set {MODELS_ENV} in .env, e.g. {MODELS_ENV}=gemini-3.1-flash-lite")
    bad = [n for n in names if not _VALID_NAME.match(n)]
    if bad:
        raise ValueError(f"not valid model names (placeholders left in .env?): {bad}")
    return names


def model_names_from_env() -> list[str]:
    return validate_model_names([n.strip() for n in os.environ.get(MODELS_ENV, "").split(",") if n.strip()])


def is_fallback_error(exc: BaseException) -> bool:
    """True for errors another model might not have: rate limits, size limits, outages.

    Checks the whole exception chain, since frameworks sometimes wrap the provider's error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "code"):
            value = getattr(current, attr, None)
            if isinstance(value, int) and value in _FALLBACK_STATUS:
                return True
        if any(hint in type(current).__name__ for hint in _FALLBACK_NAME_HINTS):
            return True
        current = current.__cause__ or current.__context__
    return False


_OVERLOAD_STATUS = {503, 529}
_OVERLOAD_HINTS = ("overloaded", "high demand", "unavailable", "try again later")


def is_overloaded(exc: BaseException) -> bool:
    """A temporary provider overload (503 "high demand", 529 "overloaded"): worth waiting a few
    seconds and retrying the same model. Rate limits and quotas (429) are not: they fall back."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "code"):
            value = getattr(current, attr, None)
            if isinstance(value, int):
                if value == 429:
                    return False
                if value in _OVERLOAD_STATUS:
                    return True
        text = str(current).lower()
        if "429" in text or "resource_exhausted" in text or "quota" in text:
            return False
        if "ServiceUnavailable" in type(current).__name__ or any(h in text for h in _OVERLOAD_HINTS):
            return True
        current = current.__cause__ or current.__context__
    return False


def strip_thoughts(request: LlmRequest) -> None:
    """Drop the model's earlier reasoning ("thought" parts) from the history, in place.

    Reasoning models return their thinking alongside tool calls, and ADK replays it
    on the next turn; some providers reject it as input (Groq: "reasoning_content is
    unsupported"). Tool calls, tool results and text are kept.
    """
    kept = []
    for content in request.contents:
        if content.role == "model" and content.parts:
            content.parts = [p for p in content.parts if not p.thought]
            if not content.parts:
                continue
        kept.append(content)
    request.contents = kept


class ThoughtlessLiteLlm(LiteLlm):
    """LiteLlm that never sends earlier reasoning back to the provider."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        request = llm_request.model_copy(deep=True)
        strip_thoughts(request)
        async for response in super().generate_content_async(request, stream=stream):
            yield response


def make_llm(name: str) -> BaseLlm:
    """Native Gemini for gemini models (keeps its thought signatures intact), LiteLLM otherwise."""
    if name.startswith("gemini/"):
        name = name.removeprefix("gemini/")
    if name.startswith("gemini"):
        from google.adk.models.google_llm import Gemini

        return Gemini(model=name)
    return ThoughtlessLiteLlm(model=name)
