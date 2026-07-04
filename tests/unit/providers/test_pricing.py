"""Tests for pricing module."""

import pytest

from agentic_rag.providers.pricing import cost_for


class TestCostFor:
    """Test cost_for function."""

    def test_anthropic_sonnet5(self) -> None:
        """Test pricing for Claude Sonnet 5."""
        # $2/$10 per million tokens
        cost = cost_for("anthropic", "claude-sonnet-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.0, abs=0.01)

    def test_anthropic_opus(self) -> None:
        """Test pricing for Claude Opus 4.8."""
        # $5/$25 per million tokens
        cost = cost_for("anthropic", "claude-opus-4-8", 1_000_000, 1_000_000)
        assert cost == pytest.approx(30.0, abs=0.01)

    def test_anthropic_haiku(self) -> None:
        """Test pricing for Claude Haiku 4.5."""
        # $1/$5 per million tokens
        cost = cost_for("anthropic", "claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert cost == pytest.approx(6.0, abs=0.01)

    def test_openai_gpt54(self) -> None:
        """Test pricing for GPT-5.4."""
        # $2.50/$15 per million tokens
        cost = cost_for("openai", "gpt-5.4", 1_000_000, 1_000_000)
        assert cost == pytest.approx(17.5, abs=0.01)

    def test_openai_delisted_model_returns_none(self) -> None:
        """Models no longer on the published pricing page cost None, not a stale price."""
        cost = cost_for("openai", "gpt-5.1", 1_000_000, 1_000_000)
        assert cost is None

    def test_google_gemini35flash(self) -> None:
        """Test pricing for Gemini 3.5 Flash."""
        # $1.50/$9.00 per million tokens
        cost = cost_for("google", "gemini-3.5-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(10.5, abs=0.01)

    def test_google_gemini31pro(self) -> None:
        """Test pricing for Gemini 3.1 Pro."""
        # $2.00/$12.00 per million tokens
        cost = cost_for("google", "gemini-3.1-pro", 1_000_000, 1_000_000)
        assert cost == pytest.approx(14.0, abs=0.01)

    def test_ollama_always_free(self) -> None:
        """Test that Ollama always returns 0.0."""
        cost = cost_for("ollama", "llama3.1:8b", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_unknown_provider_returns_none(self) -> None:
        """Test that unknown provider returns None."""
        cost = cost_for("unknown-provider", "some-model", 1000, 1000)
        assert cost is None

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown model in known provider returns None."""
        cost = cost_for("anthropic", "unknown-model-xyz", 1000, 1000)
        assert cost is None

    def test_longest_prefix_match(self) -> None:
        """Test that longest prefix is matched."""
        # "claude-sonnet-5" should match "claude-sonnet" prefix
        cost = cost_for("anthropic", "claude-sonnet-5-something", 1_000_000, 1_000_000)
        # Should match "claude-sonnet" since it's longer than "claude"
        assert cost is not None

    def test_partial_token_amounts(self) -> None:
        """Test with partial token amounts."""
        # 500K input, 200K output of Sonnet 5
        cost = cost_for("anthropic", "claude-sonnet-5", 500_000, 200_000)
        # (500K / 1M) * $2 + (200K / 1M) * $10 = $1 + $2 = $3
        assert cost == pytest.approx(3.0, abs=0.01)

    def test_zero_tokens(self) -> None:
        """Test with zero tokens."""
        cost = cost_for("anthropic", "claude-sonnet-5", 0, 0)
        assert cost == pytest.approx(0.0, abs=0.01)

    def test_gpt5_various_models(self) -> None:
        """Test GPT-5 family prefix matching."""
        # All GPT-5.x models should match correctly
        cost_55 = cost_for("openai", "gpt-5.5", 1_000_000, 1_000_000)
        assert cost_55 == pytest.approx(35.0, abs=0.01)  # $5/$30

        cost_54 = cost_for("openai", "gpt-5.4", 1_000_000, 1_000_000)
        assert cost_54 == pytest.approx(17.5, abs=0.01)  # $2.50/$15

        cost_54mini = cost_for("openai", "gpt-5.4-mini", 1_000_000, 1_000_000)
        assert cost_54mini == pytest.approx(5.25, abs=0.01)  # $0.75/$4.50
