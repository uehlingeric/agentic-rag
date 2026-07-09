"""Token-based pricing for LLM providers.

Pricing data sourced from official provider documentation:
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (2026-07-04)
- OpenAI: https://developers.openai.com/api/docs/pricing (2026-07-04)
- Google: https://ai.google.dev/gemini-api/docs/pricing (2026-07-04)
- Ollama: Local, always free (0.0)

All prices are USD per million tokens (input/output).
Ollama provider ID returns 0.0 for all inputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


# Prices as of 2026-07-04; only models listed on the providers' current pricing
# pages appear here — cost_for returns None for anything else rather than guessing.
# Anthropic Sonnet 5 introductory pricing valid through 2026-08-31.
# Vertex AI global-endpoint pricing for Gemini matches the rates below (verified
# 2026-07-04; regional endpoints are +10%). Bedrock global.* inference profiles
# match direct-API rates (verified 2026-07-09 from the Bedrock agreement rate card
# for claude-sonnet-4-6); us.* regional profiles are +10% and are not listed, so
# completions on them report cost_usd=None.
PRICES: dict[str, dict[str, ModelPrice]] = {
    "anthropic": {
        "claude-fable": ModelPrice(10.0, 50.0),
        "claude-opus-4-8": ModelPrice(5.0, 25.0),
        "claude-sonnet-5": ModelPrice(2.0, 10.0),  # Introductory through 2026-08-31
        "claude-haiku-4-5": ModelPrice(1.0, 5.0),
        # Bedrock global inference profile; rate card matches the direct API.
        "global.anthropic.claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    },
    "openai": {
        "gpt-5.5": ModelPrice(5.0, 30.0),
        "gpt-5.4": ModelPrice(2.5, 15.0),
        "gpt-5.4-mini": ModelPrice(0.75, 4.5),
        "gpt-5.4-nano": ModelPrice(0.2, 1.25),
    },
    "google": {
        "gemini-3.5-flash": ModelPrice(1.5, 9.0),
        "gemini-3.1-pro": ModelPrice(2.0, 12.0),
        "gemini-3.1-flash-lite": ModelPrice(0.25, 1.5),
    },
}


def cost_for(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculate cost in USD using longest prefix match on model ID.

    Args:
        provider: Provider name (anthropic, openai, google, ollama).
        model: Full model ID.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost in USD, 0.0 for ollama, None if provider/model unknown.
    """
    if provider == "ollama":
        return 0.0

    if provider not in PRICES:
        return None

    provider_prices = PRICES[provider]

    # Longest prefix match: try to find the best matching model price
    best_match = None
    best_length = 0

    for prefix, price in provider_prices.items():
        if model.startswith(prefix) and len(prefix) > best_length:
            best_match = price
            best_length = len(prefix)

    if best_match is None:
        return None

    # Cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    input_cost = (input_tokens / 1_000_000) * best_match.input_per_mtok
    output_cost = (output_tokens / 1_000_000) * best_match.output_per_mtok
    return input_cost + output_cost
