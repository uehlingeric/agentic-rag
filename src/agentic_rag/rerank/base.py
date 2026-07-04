"""Reranker contract. Frozen surface between retrieval and synthesis.

A reranker takes the retriever's candidates and returns them relevance-ordered,
cut to ``top_k``, with ``rank`` reassigned 1..n and ``score``/``source_scores``
preserved from the input (rerankers reorder; they do not rescore the retrieval
signal). Implementations must never drop below ``min(top_k, len(candidates))``
results or invent chunks that were not in the input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Relevance-reorders retrieval candidates for a query.

    ``last_usage`` reports LLM tokens spent by the most recent ``rerank`` call
    (``Usage.zero()`` for non-LLM backends) so the pipeline can account for it.
    """

    name: str
    last_usage: Usage

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]: ...


class NoopReranker:
    """Pass-through: keeps retrieval order, cuts to top_k, reassigns ranks."""

    name = "none"

    def __init__(self) -> None:
        self.last_usage = Usage.zero()

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        return [replace(c, rank=i) for i, c in enumerate(candidates[:top_k], start=1)]
