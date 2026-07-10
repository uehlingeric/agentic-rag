"""Stub provider: deterministic canned answers for CI and API testing.

Zero-model, deterministic provider that exercises the full guarded pipeline
without requiring real API keys or model access. Useful for smoke tests and
CI profiles. The LLM stub returns "[1]" citation marker resolved from the first
retrieved context chunk.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import AsyncIterator, Sequence

from agentic_rag.providers.base import (
    Completion,
    EmbeddingResult,
    Message,
    StreamEvent,
    Usage,
)
from agentic_rag.tokens import count_tokens


class StubLLMProvider:
    """Deterministic LLM provider returning canned answers."""

    name: str = "stub"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Return a deterministic canned answer."""
        joined = "\n".join(m.content for m in messages)
        if system:
            joined = system + "\n" + joined

        input_tokens = count_tokens(joined)
        text = "Stub answer derived from the provided context [1]."
        output_tokens = len(text.split())

        return Completion(
            text=text,
            model="stub",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
            ),
            stop_reason="end_turn",
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream deterministic response in 2-3 deltas."""

        async def _stream() -> AsyncIterator[StreamEvent]:
            joined = "\n".join(m.content for m in messages)
            if system:
                joined = system + "\n" + joined

            input_tokens = count_tokens(joined)
            parts = ["Stub answer ", "derived from the provided context ", "[1]."]

            for part in parts:
                yield StreamEvent(delta=part)

            text = "".join(parts)
            output_tokens = len(text.split())

            yield StreamEvent(
                completion=Completion(
                    text=text,
                    model="stub",
                    usage=Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=0.0,
                    ),
                    stop_reason="end_turn",
                )
            )

        return _stream()

    def count_tokens(self, text: str) -> int:
        """Count tokens using the standard encoding."""
        return count_tokens(text)


class StubEmbeddingProvider:
    """Deterministic embedding provider returning seeded unit-norm vectors."""

    name = "stub"

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult:
        """Return deterministic unit-norm 64-dimensional embeddings."""
        vectors: list[list[float]] = []

        for text in texts:
            # Seed RNG per text using CRC32
            seed = zlib.crc32(text.encode())
            rng = random.Random(seed)

            # Generate 64-dim random vector
            vec = [rng.gauss(0, 1) for _ in range(64)]

            # Normalize to unit norm
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]

            vectors.append(vec)

        return EmbeddingResult(
            vectors=vectors,
            model="stub",
            dimensions=64,
            usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0),
        )
