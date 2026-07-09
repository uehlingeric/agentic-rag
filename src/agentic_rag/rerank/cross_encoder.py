"""Local cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ScoredChunk


class CrossEncoderReranker:
    """Reranks candidates using a local cross-encoder model.

    Requires the sentence-transformers package (install via
    ``uv sync --extra rerank-local``). The model is lazy-loaded on first
    rerank call.
    """

    name = "cross-encoder"

    def __init__(self, *, model: str | None = None) -> None:
        """Initialize the reranker with an optional model name.

        Args:
            model: Model identifier for sentence-transformers CrossEncoder.
                Defaults to "BAAI/bge-reranker-base".
        """
        self._model_name = model or "BAAI/bge-reranker-base"
        self._model: Any | None = None
        self.last_usage = Usage.zero()

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        """Rerank candidates by cross-encoder relevance score.

        Scores (query, chunk_text) pairs, sorts descending by score (stable on
        ties), cuts to top_k, and reassigns ranks 1..n. Preserves original
        scores and source_scores.

        Args:
            query: The query string.
            candidates: Candidate chunks to rerank.
            top_k: Maximum number of results to return.

        Returns:
            Reranked candidates, cut to top_k, with ranks reassigned 1..n.
        """
        if not candidates:
            self.last_usage = Usage.zero()
            return []

        # Lazy-load the model on first use
        if self._model is None:
            self._model = self._load_model()

        # Build (query, chunk_text) pairs for all candidates
        pairs = [(query, candidate.chunk.text) for candidate in candidates]

        # Score via asyncio.to_thread (predict is sync/CPU-bound)
        scores = await asyncio.to_thread(self._model.predict, pairs)

        # Convert scores to float (robustness against numpy arrays, etc.)
        scores = [float(s) for s in scores]

        # Sort by score descending (stable sort preserves input order on ties)
        scored_candidates = list(zip(candidates, scores, strict=True))
        scored_candidates.sort(key=lambda x: -x[1])  # Negative for descending

        # Cut to top_k and reassign ranks 1..n, preserving score/source_scores
        result = [
            replace(candidate, rank=i)
            for i, (candidate, _score) in enumerate(scored_candidates[:top_k], start=1)
        ]

        self.last_usage = Usage.zero()
        return result

    def _load_model(self) -> Any:
        """Load the CrossEncoder model lazily.

        Raises ImportError if sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            msg = (
                "sentence-transformers is not installed. "
                "Install it with: uv sync --extra rerank-local"
            )
            raise ImportError(msg) from e

        return CrossEncoder(self._model_name)
