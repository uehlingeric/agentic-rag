"""Tests for stub provider: deterministic canned answers."""

from __future__ import annotations

from agentic_rag.providers.base import Message, Role
from agentic_rag.providers.stub import StubEmbeddingProvider, StubLLMProvider


class TestStubLLMProvider:
    """Tests for StubLLMProvider."""

    async def test_complete_returns_canned_answer(self) -> None:
        """complete() returns deterministic answer with [1] marker."""
        provider = StubLLMProvider()
        messages = [Message(role=Role.USER, content="What is this?")]

        completion = await provider.complete(messages)

        assert completion.model == "stub"
        assert "[1]" in completion.text
        assert completion.usage.input_tokens > 0
        assert completion.usage.output_tokens > 0
        assert completion.usage.cost_usd == 0.0
        assert completion.stop_reason == "end_turn"

    async def test_complete_with_system_prompt(self) -> None:
        """complete() includes system prompt in input tokens."""
        provider = StubLLMProvider()
        messages = [Message(role=Role.USER, content="Test")]
        system = "You are helpful."

        completion = await provider.complete(messages, system=system)

        assert completion.usage.input_tokens > 0

    async def test_stream_yields_deltas_and_completion(self) -> None:
        """stream() yields delta events then terminal completion event."""
        provider = StubLLMProvider()
        messages = [Message(role=Role.USER, content="What is this?")]

        events = [e async for e in provider.stream(messages)]

        # Should have at least 3 deltas + 1 completion
        assert len(events) >= 4

        # Last event has completion
        assert events[-1].completion is not None
        assert events[-1].completion.model == "stub"
        assert "[1]" in events[-1].completion.text

        # Earlier events have deltas
        delta_events = [e for e in events[:-1] if e.delta]
        assert len(delta_events) > 0


class TestStubEmbeddingProvider:
    """Tests for StubEmbeddingProvider."""

    async def test_embed_batch_deterministic(self) -> None:
        """embed_batch() returns same vectors for same text."""
        provider = StubEmbeddingProvider()

        result1 = await provider.embed_batch(["Hello world"])
        result2 = await provider.embed_batch(["Hello world"])

        assert result1.vectors[0] == result2.vectors[0]

    async def test_embed_batch_different_text(self) -> None:
        """embed_batch() returns different vectors for different text."""
        provider = StubEmbeddingProvider()

        result1 = await provider.embed_batch(["Hello"])
        result2 = await provider.embed_batch(["World"])

        assert result1.vectors[0] != result2.vectors[0]

    async def test_embed_batch_unit_norm(self) -> None:
        """embed_batch() returns unit-norm vectors."""
        provider = StubEmbeddingProvider()

        result = await provider.embed_batch(["Test"])

        for vec in result.vectors:
            norm = sum(x * x for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-6  # Unit norm ≈ 1.0

    async def test_embed_batch_64_dimensions(self) -> None:
        """embed_batch() returns 64-dimensional vectors."""
        provider = StubEmbeddingProvider()

        result = await provider.embed_batch(["Test"])

        assert result.dimensions == 64
        assert len(result.vectors[0]) == 64

    async def test_embed_batch_zero_cost(self) -> None:
        """embed_batch() reports zero cost."""
        provider = StubEmbeddingProvider()

        result = await provider.embed_batch(["Test"])

        assert result.usage.cost_usd == 0.0
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
