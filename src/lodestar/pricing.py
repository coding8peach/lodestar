"""Estimate what model calls cost.

Prices come from config/prices.yaml when it names the model, otherwise from LiteLLM's
built-in price table. A model with no known price gets None, never 0, so a missing
price stays visible. Two numbers per call:
  - list cost: tokens x list price, what the call would cost on a paid plan
  - cost:      what you actually pay; 0 for models marked free_tier in prices.yaml
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from lodestar.paths import PROJECT_ROOT
from lodestar.schemas.fit import LlmCall

log = logging.getLogger(__name__)

PRICES_FILE = PROJECT_ROOT / "config" / "prices.yaml"


@dataclass(frozen=True)
class Price:
    input_per_mtok: float    # USD per million input tokens
    output_per_mtok: float   # USD per million output tokens
    source: str              # "prices.yaml" or "litellm"
    free_tier: bool = False


@lru_cache(maxsize=1)
def _overrides(path: Path = PRICES_FILE) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _litellm_price(model: str) -> tuple[float, float] | None:
    try:
        import litellm
    except ImportError:
        return None
    bare = model.split("/", 1)[1] if model.startswith(("openai/", "gemini/")) else model
    for key in (model, bare, f"gemini/{bare}"):
        entry = litellm.model_cost.get(key)
        if entry and entry.get("input_cost_per_token") is not None and entry.get("output_cost_per_token") is not None:
            return entry["input_cost_per_token"] * 1e6, entry["output_cost_per_token"] * 1e6
    return None


def price_for(model: str, overrides: dict | None = None) -> Price | None:
    overrides = _overrides() if overrides is None else overrides
    entry = overrides.get(model) or {}
    free = bool(entry.get("free_tier", False))
    if "input_per_mtok" in entry and "output_per_mtok" in entry:
        return Price(float(entry["input_per_mtok"]), float(entry["output_per_mtok"]), "prices.yaml", free)
    table = _litellm_price(model)
    if table is None:
        return None
    return Price(table[0], table[1], "litellm", free)


def estimate(call: LlmCall, overrides: dict | None = None) -> tuple[float | None, float | None, str | None]:
    """(list cost USD, cost you pay USD, price source) for one call."""
    price = price_for(call.model, overrides)
    if price is None:
        return None, None, None
    list_cost = (call.input_tokens * price.input_per_mtok + call.output_tokens * price.output_per_mtok) / 1e6
    return list_cost, (0.0 if price.free_tier else list_cost), price.source
