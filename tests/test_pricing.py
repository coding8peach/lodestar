from datetime import datetime, timezone

from lodestar.pricing import estimate, price_for
from lodestar.schemas import LlmCall


def call(model="m", tokens_in=1_000_000, tokens_out=100_000) -> LlmCall:
    return LlmCall(model=model, attempt=1, status="ok", input_tokens=tokens_in, output_tokens=tokens_out,
                   started_at=datetime(2026, 9, 28, tzinfo=timezone.utc))


def test_price_from_overrides():
    overrides = {"m": {"input_per_mtok": 0.5, "output_per_mtok": 2.0}}
    assert estimate(call(), overrides) == (0.7, 0.7, "prices.yaml")  # 1M x 0.5 + 0.1M x 2.0


def test_free_tier_pays_nothing_but_keeps_list_cost():
    overrides = {"m": {"input_per_mtok": 0.5, "output_per_mtok": 2.0, "free_tier": True}}
    assert estimate(call(), overrides) == (0.7, 0.0, "prices.yaml")


def test_litellm_table_used_when_not_overridden():
    price = price_for("openai/gpt-5.4-nano", overrides={})
    assert price is not None and price.source == "litellm" and price.input_per_mtok > 0


def test_free_tier_flag_with_litellm_price():
    price = price_for("gemini-3.1-flash-lite", overrides={"gemini-3.1-flash-lite": {"free_tier": True}})
    assert price.free_tier and price.source == "litellm"


def test_unknown_model_has_no_price_not_zero():
    assert price_for("made-up-model", overrides={}) is None
    assert estimate(call("made-up-model"), overrides={}) == (None, None, None)
