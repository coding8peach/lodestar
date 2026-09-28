"""Scripted stand-ins shared by tests: no real LLM calls."""

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class RateLimited(Exception):
    status_code = 429


class FakeLlm(BaseLlm):
    """Replies from a script: each step is an Exception to raise or a list of Parts to return.
    Records every request it receives."""

    script: list = []
    requests: list = []
    usage: tuple[int, int, int] | None = None  # (prompt, output, thinking) tokens reported per response

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.requests.append(llm_request.model_copy(deep=True))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        usage = None
        if self.usage:
            prompt, output, thinking = self.usage
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thinking,
                total_token_count=prompt + output + thinking)
        yield LlmResponse(content=types.Content(role="model", parts=step), usage_metadata=usage)


def text(t: str) -> list:
    return [types.Part(text=t)]


def tool_calls(job_id: str) -> list:
    return [
        types.Part(function_call=types.FunctionCall(name="get_job", args={"job_id": job_id})),
        types.Part(function_call=types.FunctionCall(name="get_profile", args={})),
    ]
