"""Ask the fit agent service for an analysis over A2A.

This is all a caller needs: it doesn't import ADK, MCP toolsets or model configuration.
The service's address comes from LODESTAR_AGENT_URL (default http://localhost:8001).
"""

import os

import httpx
from a2a.client import A2AClientError, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import a2a_pb2
from pydantic import ValidationError

from lodestar.schemas.fit import FitAgentReply
from lodestar.schemas.resume import ResumeAgentReply

DEFAULT_URL = "http://localhost:8001"
TIMEOUT_SECONDS = 300.0  # an analysis can take a while, especially when models fall back


class FitAgentUnavailable(RuntimeError):
    """The service couldn't be reached (not started, wrong URL)."""


class FitAgentError(RuntimeError):
    """The service answered, but the analysis failed; the message is the service's."""


def agent_url() -> str:
    return os.environ.get("LODESTAR_AGENT_URL", DEFAULT_URL).rstrip("/")


DEFAULT_RESUME_URL = "http://localhost:8002"


def resume_agent_url() -> str:
    return os.environ.get("LODESTAR_RESUME_AGENT_URL", DEFAULT_RESUME_URL).rstrip("/")


async def _send(job_id: str, base_url: str, httpx_client: httpx.AsyncClient | None,
                max_paid_usd: float | None):
    """Send a job_id to an agent service; return the final A2A response."""
    own_client = httpx_client is None
    http = httpx_client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        try:
            client = await create_client(base_url, client_config=ClientConfig(httpx_client=http, streaming=False))
        except (httpx.ConnectError, A2AClientError) as e:
            raise FitAgentUnavailable(
                f"agent not reachable at {base_url}; start it with `uv run lodestar-agent` ({type(e).__name__})"
            ) from e
        request = a2a_pb2.SendMessageRequest(message=new_text_message(job_id, role=a2a_pb2.Role.ROLE_USER))
        if max_paid_usd is not None:
            request.metadata.update({"max_paid_usd": max_paid_usd})
        last = None
        try:
            async for response in client.send_message(request):
                last = response
        except httpx.ConnectError as e:
            raise FitAgentUnavailable(f"lost the connection to the agent at {base_url}") from e
        return last
    finally:
        if own_client:
            await http.aclose()


async def analyze_via_a2a(
    job_id: str,
    base_url: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    max_paid_usd: float | None = None,
) -> FitAgentReply:
    """`max_paid_usd`: how much paid-model spend this analysis may use. 0 means free models
    only; None means no limit (the service's full model list)."""
    last = await _send(job_id, base_url or agent_url(), httpx_client, max_paid_usd)
    return _parse(FitAgentReply, _reply_text(last))


async def tailor_via_a2a(
    job_id: str,
    base_url: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    max_paid_usd: float | None = None,
) -> ResumeAgentReply:
    """Ask the resume agent for a tailored resume draft (checked against the profile)."""
    last = await _send(job_id, base_url or resume_agent_url(), httpx_client, max_paid_usd)
    return _parse(ResumeAgentReply, _reply_text(last))


def _parse(model, text: str):
    try:
        return model.model_validate_json(text)
    except ValidationError as e:
        raise FitAgentError(f"the agent's reply isn't a {model.__name__}: {e.errors()[0]['msg']}") from e


def _reply_text(response) -> str:
    """The completed task's text, or FitAgentError with the service's message."""
    if response is None or not response.HasField("task"):
        raise FitAgentError("the agent returned no task")
    task = response.task
    state = a2a_pb2.TaskState.Name(task.status.state)
    if state != "TASK_STATE_COMPLETED":
        detail = get_message_text(task.status.message) if task.status.HasField("message") else ""
        raise FitAgentError(detail or f"agent task ended as {state}")
    return "\n".join(get_artifact_text(a) for a in task.artifacts).strip()


def agent_status(base_url: str | None = None, timeout: float = 2.0) -> tuple[bool, str]:
    """(running, message): a quick check that the fit agent service answers its agent card."""
    base_url = base_url or agent_url()
    try:
        resp = httpx.get(f"{base_url}/.well-known/agent-card.json", timeout=timeout)
        resp.raise_for_status()
        return True, f"fit agent running at {base_url}"
    except httpx.HTTPError:
        return False, f"fit agent not running at {base_url}; start it with `uv run lodestar-agent`"
