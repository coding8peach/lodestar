"""Project paths, independent of the current working directory.

Services may be launched from elsewhere (e.g. the fit agent starts the MCP
server as a subprocess), so defaults are anchored to the project root.
Assumes the editable install that `uv` sets up for this project.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def db_path() -> Path:
    return Path(os.environ.get("LODESTAR_DB", DATA_DIR / "lodestar.sqlite"))


def profile_path() -> Path:
    return Path(os.environ.get("LODESTAR_PROFILE", DATA_DIR / "profile.yaml"))
