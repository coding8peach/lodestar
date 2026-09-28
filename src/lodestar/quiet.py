"""Keep terminal output readable: our own logs, not framework noise."""

import logging
import warnings


def silence() -> None:
    # ADK announces experimental features and deprecations on every run.
    warnings.filterwarnings("ignore", message=r".*\[EXPERIMENTAL\].*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\..*")
    # ADK logs full tracebacks for errors we already catch and report in one line.
    for name in ("google_adk", "google.adk", "google_genai", "httpx", "mcp", "a2a"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    for name in ("LiteLLM", "litellm"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        import litellm

        litellm.suppress_debug_info = True  # the "Give Feedback / Get Help" and "Provider List" banners
    except ImportError:
        pass
