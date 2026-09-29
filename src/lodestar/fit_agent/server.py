"""Serve Lodestar's agents over A2A, in one process:

    uv run lodestar-agent
      fit agent     http://localhost:8001   (LODESTAR_AGENT_PORT)
      resume agent  http://localhost:8002   (LODESTAR_RESUME_AGENT_PORT)

Port: LODESTAR_AGENT_PORT (default 8001). The same port is passed to to_a2a, which
puts it in the agent card, and to uvicorn, which listens on it; they must match.
Models: LODESTAR_FIT_MODELS, read here; choosing models is this service's business.
Agent card: http://localhost:<port>/.well-known/agent-card.json
"""

import logging
import os
import sys

from lodestar.quiet import silence

silence()

import uvicorn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from google.adk.a2a.utils.agent_to_a2a import to_a2a  # noqa: E402

from lodestar.fit_agent.models import model_names_from_env  # noqa: E402
from lodestar.fit_agent.service import build_fit_service  # noqa: E402
from lodestar.paths import PROJECT_ROOT  # noqa: E402

log = logging.getLogger("lodestar.fit_agent.server")

DEFAULT_PORT = 8001
HOST = "localhost"


def agent_port() -> int:
    return int(os.environ.get("LODESTAR_AGENT_PORT", DEFAULT_PORT))


def resume_agent_port() -> int:
    return int(os.environ.get("LODESTAR_RESUME_AGENT_PORT", 8002))


def build_resume_app(model_names: list[str], port: int, **service_kwargs):
    from lodestar.resume_agent.service import build_resume_service

    app = to_a2a(build_resume_service(model_names, **service_kwargs), host=HOST, port=port)
    silence()
    return app


def build_app(model_names: list[str], port: int, **service_kwargs):
    app = to_a2a(build_fit_service(model_names, **service_kwargs), host=HOST, port=port)
    silence()  # to_a2a turns ADK's logger back up to INFO; failed tasks would log full tracebacks
    return app


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    silence()
    try:
        models = model_names_from_env()
    except ValueError as e:
        log.error(str(e))
        return 2
    import asyncio

    fit_port, resume_port = agent_port(), resume_agent_port()
    servers = [
        uvicorn.Server(uvicorn.Config(build_app(models, fit_port), host=HOST, port=fit_port, log_level="warning")),
        uvicorn.Server(uvicorn.Config(build_resume_app(models, resume_port), host=HOST, port=resume_port,
                                      log_level="warning")),
    ]
    log.info("fit agent on http://%s:%d, resume agent on http://%s:%d (models: %s)",
             HOST, fit_port, HOST, resume_port, ", ".join(models))

    async def serve_both() -> None:
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(serve_both())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
