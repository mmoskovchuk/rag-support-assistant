"""Token usage and cost of every LLM call, so spending is visible in the logs and API responses."""

import logging
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pricing":
        return cls(settings.openai_input_price_per_1m, settings.openai_output_price_per_1m)


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


def record_usage(message: BaseMessage, pricing: Pricing, purpose: str) -> LLMUsage | None:
    """Log and return the usage of one LLM call; None if the model did not report it (e.g. test fakes)."""
    meta = getattr(message, "usage_metadata", None)
    if not meta:
        return None
    usage = LLMUsage(
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
        cost_usd=round(
            (meta["input_tokens"] * pricing.input_per_1m + meta["output_tokens"] * pricing.output_per_1m) / 1_000_000,
            6,
        ),
    )
    logger.info(
        "LLM usage (%s): %d in + %d out tokens = $%.6f",
        purpose,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    return usage
