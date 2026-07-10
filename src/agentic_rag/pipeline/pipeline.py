"""RAG pipeline: retrieve -> rerank -> synthesize, with citation validation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentic_rag.config import Settings
from agentic_rag.pipeline.base import Answer, StageTiming, scrub_sentinel
from agentic_rag.pipeline.citations import resolve_citations
from agentic_rag.pipeline.context import build_context
from agentic_rag.pipeline.synthesize import stream_synthesis, synthesize
from agentic_rag.providers.base import LLMProvider
from agentic_rag.rerank.base import Reranker
from agentic_rag.retrieval.base import RetrievalMode, ScoredChunk


@runtime_checkable
class SupportsRetrieve(Protocol):
    """Protocol for retrieval interface matching Retriever.retrieve signature."""

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Retrieve chunks matching the query."""
        ...


@dataclass(frozen=True, slots=True)
class AskStreamEvent:
    """Event from streaming ask() call.

    Text events carry ``delta``; the single terminal event carries the
    assembled ``answer`` and an empty delta.
    """

    delta: str = ""
    answer: Answer | None = None


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    Orchestrates retrieve -> rerank -> context build -> synthesize -> citations.
    """

    def __init__(
        self, retriever: SupportsRetrieve, reranker: Reranker, llm: LLMProvider, settings: Settings
    ) -> None:
        """Initialize the pipeline.

        Args:
            retriever: Retrieval interface (retrieve chunks by query).
            reranker: Reranker interface (relevance-order and top-k cut).
            llm: LLM provider for synthesis and streaming.
            settings: Application settings (context/answer token budgets).
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.settings = settings

    async def ask(self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID) -> Answer:
        """Answer a question from the retrieval corpus.

        Retrieves candidates, reranks, builds context, synthesizes with LLM,
        validates citations, and returns the complete Answer.

        Args:
            question: The user's question.
            mode: Retrieval mode (BM25, dense, or hybrid).

        Returns:
            Answer with text, citations, context, usage, and timings.
        """
        # Retrieve: fetch candidate_pool from retriever
        start_retrieve = time.perf_counter()
        candidates = await self.retriever.retrieve(
            question, mode=mode, top_k=self.settings.rerank.candidate_pool
        )
        elapsed_retrieve = time.perf_counter() - start_retrieve

        # Rerank: cut to top_k and capture usage
        start_rerank = time.perf_counter()
        reranked = await self.reranker.rerank(
            question, candidates, top_k=self.settings.rerank.top_k
        )
        elapsed_rerank = time.perf_counter() - start_rerank
        rerank_usage = self.reranker.last_usage

        # Build context: format and budget chunks
        built = build_context(
            reranked,
            max_tokens=self.settings.synthesis.max_context_tokens,
            count_tokens=self.llm.count_tokens,
        )

        # Synthesize: LLM answer generation
        start_synthesis = time.perf_counter()
        synth_result = await synthesize(
            self.llm, question, built, max_tokens=self.settings.synthesis.max_answer_tokens
        )
        elapsed_synthesis = time.perf_counter() - start_synthesis

        # Citations: resolve markers and validate
        citation_result = resolve_citations(synth_result.text, built.chunks)

        # Assemble Answer
        return Answer(
            text=citation_result.text,
            citations=citation_result.citations,
            context=built.chunks,
            usage=rerank_usage + synth_result.usage,
            timings=[
                StageTiming("retrieve", elapsed_retrieve),
                StageTiming("rerank", elapsed_rerank),
                StageTiming("synthesize", elapsed_synthesis),
            ],
            refusal=synth_result.refusal,
            invalid_citations=citation_result.invalid_markers,
        )

    async def ask_stream(
        self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> AsyncIterator[AskStreamEvent]:
        """Stream answer to a question, buffering the refusal sentinel.

        Retrieves, reranks, builds context, then streams synthesis while
        buffering and stripping the refusal sentinel. Yields text deltas
        and a single terminal event with the complete Answer.

        Args:
            question: The user's question.
            mode: Retrieval mode (BM25, dense, or hybrid).

        Yields:
            AskStreamEvent with delta (text events) or answer (terminal).
        """
        # Retrieve: fetch candidate_pool from retriever
        start_retrieve = time.perf_counter()
        candidates = await self.retriever.retrieve(
            question, mode=mode, top_k=self.settings.rerank.candidate_pool
        )
        elapsed_retrieve = time.perf_counter() - start_retrieve

        # Rerank: cut to top_k and capture usage
        start_rerank = time.perf_counter()
        reranked = await self.reranker.rerank(
            question, candidates, top_k=self.settings.rerank.top_k
        )
        elapsed_rerank = time.perf_counter() - start_rerank
        rerank_usage = self.reranker.last_usage

        # Build context: format and budget chunks
        built = build_context(
            reranked,
            max_tokens=self.settings.synthesis.max_context_tokens,
            count_tokens=self.llm.count_tokens,
        )

        # Stream synthesis with sentinel buffering
        start_synthesis = time.perf_counter()
        async for event in stream_synthesis(
            self.llm, question, built, max_tokens=self.settings.synthesis.max_answer_tokens
        ):
            # Text delta event
            if event.completion is None:
                yield AskStreamEvent(delta=event.delta)
            else:
                # Terminal event: post-process and resolve citations
                elapsed_synthesis = time.perf_counter() - start_synthesis

                scrub = scrub_sentinel(event.completion.text)

                citation_result = resolve_citations(scrub.text, built.chunks)

                answer = Answer(
                    text=citation_result.text,
                    citations=citation_result.citations,
                    context=built.chunks,
                    usage=rerank_usage + event.completion.usage,
                    timings=[
                        StageTiming("retrieve", elapsed_retrieve),
                        StageTiming("rerank", elapsed_rerank),
                        StageTiming("synthesize", elapsed_synthesis),
                    ],
                    refusal=scrub.refusal,
                    invalid_citations=citation_result.invalid_markers,
                )

                yield AskStreamEvent(answer=answer)
                return
