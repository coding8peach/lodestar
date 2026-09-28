"""Serve the fit agent over A2A.

    uv run lodestar-agent

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
    port = agent_port()
    app = build_app(models, port)
    log.info("fit agent on http://%s:%d (models: %s)", HOST, port, ", ".join(models))
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
