"""Tests for multi-query gather with reranking, deduplication, and budgeting."""

from __future__ import annotations

import pytest

from agentic_rag.agent.gather import gather
from agentic_rag.agent.state import Plan, PlanKind
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    heading: str = "AC-2 ACCOUNT MANAGEMENT",
    text: str = "The organization manages system accounts.",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=heading,
        heading=heading,
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_scored_chunk(chunk: ChunkRecord, score: float = 0.9, rank: int = 1) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


class StubRetriever:
    """Stub retriever that returns canned results per query."""

    def __init__(self) -> None:
        self.retrieved_queries: list[tuple[str, RetrievalMode, int]] = []
        self.results: dict[str, list[ScoredChunk]] = {}

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Record the call and return canned results."""
        self.retrieved_queries.append((query, mode, top_k))
        return self.results.get(query, [])


class StubReranker:
    """Stub reranker that records calls, sets last_usage, and cuts to top_k."""

    def __init__(self) -> None:
        self.name = "stub-reranker"
        self.last_usage = Usage.zero()
        self.reranked_calls: list[tuple[str, int]] = []
        self.usages: list[Usage] = []

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        """Record the call, set last_usage, and return top_k."""
        self.reranked_calls.append((query, top_k))

        # Pop the next usage from the list
        if self.usages:
            self.last_usage = self.usages.pop(0)
        else:
            self.last_usage = Usage(input_tokens=10, output_tokens=5, cost_usd=0.001)

        # Cut to top_k and reassign ranks
        from dataclasses import replace

        return [replace(c, rank=i) for i, c in enumerate(candidates[:top_k], start=1)]


def count_tokens(text: str) -> int:
    """Simple token counter: split on whitespace."""
    return len(text.split())


@pytest.mark.asyncio
async def test_two_sub_queries_retrieved_and_reranked() -> None:
    """Two sub-queries are both retrieved and reranked with correct args."""
    chunk_a = make_chunk("c-a", text="Account management content")
    chunk_b = make_chunk("c-b", text="Audit logging content")
    chunk_c = make_chunk("c-c", text="Risk management content")
    chunk_d = make_chunk("c-d", text="Crypto standards content")

    retriever = StubRetriever()
    retriever.results["What is account management?"] = [
        make_scored_chunk(chunk_a, score=0.95),
        make_scored_chunk(chunk_b, score=0.80),
    ]
    retriever.results["What is risk governance?"] = [
        make_scored_chunk(chunk_c, score=0.92),
        make_scored_chunk(chunk_d, score=0.85),
    ]

    reranker = StubReranker()
    reranker.usages = [
        Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
        Usage(input_tokens=25, output_tokens=12, cost_usd=0.0025),
    ]

    plan = Plan(
        kind=PlanKind.MULTI_HOP,
        sub_queries=("What is account management?", "What is risk governance?"),
    )

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=2,
        max_context_tokens=1000,
        count_tokens=count_tokens,
    )

    # Verify both sub-queries were retrieved with correct arguments
    assert len(retriever.retrieved_queries) == 2
    assert retriever.retrieved_queries[0] == (
        "What is account management?",
        RetrievalMode.HYBRID,
        10,
    )
    assert retriever.retrieved_queries[1] == ("What is risk governance?", RetrievalMode.HYBRID, 10)

    # Verify both sub-queries were reranked with correct top_k
    assert len(reranker.reranked_calls) == 2
    assert reranker.reranked_calls[0] == ("What is account management?", 2)
    assert reranker.reranked_calls[1] == ("What is risk governance?", 2)

    # Verify sub_results
    assert len(result.sub_results) == 2
    assert result.sub_results[0].query == "What is account management?"
    assert result.sub_results[1].query == "What is risk governance?"

    # Verify usage is summed
    assert result.usage.input_tokens == 45
    assert result.usage.output_tokens == 22
    assert result.usage.cost_usd is not None
    assert abs(result.usage.cost_usd - 0.0045) < 1e-10


@pytest.mark.asyncio
async def test_dedupe_shared_chunk() -> None:
    """Chunk shared by both sub-queries appears once; second query still gets its other chunks."""
    chunk_shared = make_chunk("c-shared", text="Shared content")
    chunk_a1 = make_chunk("c-a1", text="Query A specific")
    chunk_a2 = make_chunk("c-a2", text="Query A other")
    chunk_b1 = make_chunk("c-b1", text="Query B specific")

    retriever = StubRetriever()
    retriever.results["Query A"] = [
        make_scored_chunk(chunk_a1, score=0.95),
        make_scored_chunk(chunk_shared, score=0.85),
        make_scored_chunk(chunk_a2, score=0.80),
    ]
    retriever.results["Query B"] = [
        make_scored_chunk(chunk_shared, score=0.88),
        make_scored_chunk(chunk_b1, score=0.82),
    ]

    reranker = StubReranker()
    reranker.usages = [
        Usage.zero(),
        Usage.zero(),
    ]

    plan = Plan(kind=PlanKind.MULTI_HOP, sub_queries=("Query A", "Query B"))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=3,
        max_context_tokens=1000,
        count_tokens=count_tokens,
    )

    # sub_results carry each sub-query's full post-rerank list (pre-merge)
    assert len(result.sub_results[0].chunks) == 3
    assert result.sub_results[0].chunks[0].chunk.chunk_id == "c-a1"
    assert result.sub_results[0].chunks[1].chunk.chunk_id == "c-shared"
    assert result.sub_results[0].chunks[2].chunk.chunk_id == "c-a2"
    assert len(result.sub_results[1].chunks) == 2
    assert result.sub_results[1].chunks[0].chunk.chunk_id == "c-shared"
    assert result.sub_results[1].chunks[1].chunk.chunk_id == "c-b1"

    # Merged context dedupes c-shared: a1, shared, a2, b1 — and B still
    # contributed its other chunk (dedupe neither stopped its walk nor charged budget)
    assert [s.chunk.chunk_id for s in result.context.chunks] == ["c-a1", "c-shared", "c-a2", "c-b1"]


@pytest.mark.asyncio
async def test_proportional_budget_respects_per_query_limit() -> None:
    """A chunk over its sub-query's budget is excluded even with global room left.

    With the whitespace counter every excerpt is exactly 20 tokens: 9 tokens
    for the <excerpt id=n source="..."> line, 10 text tokens, and 1 for the
    closing tag. max_context_tokens=90 gives each of the two sub-queries a
    45-token budget: two excerpts (40) fit, a third (60) does not — so each
    query contributes exactly 2 chunks and the merged total (80 tokens) stays
    under the untouched global budget of 90.
    """
    chunk_1 = make_chunk("c-1", text="word " * 10)
    chunk_2 = make_chunk("c-2", text="word " * 10)
    chunk_3 = make_chunk("c-3", text="word " * 10)
    chunk_4 = make_chunk("c-4", text="word " * 10)
    chunk_5 = make_chunk("c-5", text="word " * 10)
    chunk_6 = make_chunk("c-6", text="word " * 10)

    retriever = StubRetriever()
    retriever.results["Query A"] = [
        make_scored_chunk(chunk_1),
        make_scored_chunk(chunk_2),
        make_scored_chunk(chunk_3),
    ]
    retriever.results["Query B"] = [
        make_scored_chunk(chunk_4),
        make_scored_chunk(chunk_5),
        make_scored_chunk(chunk_6),
    ]

    reranker = StubReranker()
    reranker.usages = [Usage.zero(), Usage.zero()]

    plan = Plan(kind=PlanKind.MULTI_HOP, sub_queries=("Query A", "Query B"))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=3,
        max_context_tokens=90,  # budget_per_query = 45
        count_tokens=count_tokens,
    )

    assert [s.chunk.chunk_id for s in result.context.chunks] == ["c-1", "c-2", "c-4", "c-5"]
    assert result.context.token_count == 80  # 4 excerpts x 20 tokens, under the global 90


@pytest.mark.asyncio
async def test_first_chunk_exception_exceeds_budget() -> None:
    """Sub-query's first new chunk is always included even if it exceeds budget."""
    # Create a very large chunk
    large_chunk = make_chunk("c-large", text="word " * 100)  # 500 tokens
    small_chunk = make_chunk("c-small", text="word")  # 1 token

    retriever = StubRetriever()
    retriever.results["Query"] = [
        make_scored_chunk(large_chunk),
        make_scored_chunk(small_chunk),
    ]

    reranker = StubReranker()
    reranker.usages = [Usage.zero()]

    plan = Plan(kind=PlanKind.DIRECT, sub_queries=("Query",))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=2,
        max_context_tokens=50,  # Very small budget per query
        count_tokens=count_tokens,
    )

    # First chunk should be included despite exceeding budget
    assert len(result.sub_results[0].chunks) >= 1
    assert result.sub_results[0].chunks[0].chunk.chunk_id == "c-large"


@pytest.mark.asyncio
async def test_single_sub_query_equals_direct_retrieve_rerank() -> None:
    """DIRECT plan produces identical context to plain retrieve -> rerank -> build_context."""
    from agentic_rag.pipeline.context import build_context

    chunk_a = make_chunk("c-a", text="Account management")
    chunk_b = make_chunk("c-b", text="Audit logging")
    chunk_c = make_chunk("c-c", text="Risk management")

    retriever = StubRetriever()
    retriever.results["Question"] = [
        make_scored_chunk(chunk_a, score=0.95),
        make_scored_chunk(chunk_b, score=0.85),
        make_scored_chunk(chunk_c, score=0.75),
    ]

    reranker = StubReranker()
    reranker.usages = [Usage.zero()]

    plan = Plan(kind=PlanKind.DIRECT, sub_queries=("Question",))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=2,
        max_context_tokens=1000,
        count_tokens=count_tokens,
    )

    # Compare with direct call to build_context
    reranked_chunks = [
        make_scored_chunk(chunk_a, score=0.95, rank=1),
        make_scored_chunk(chunk_b, score=0.85, rank=2),
    ]
    direct_context = build_context(
        reranked_chunks,
        max_tokens=1000,
        count_tokens=count_tokens,
    )

    # The gathered context should match the direct build
    assert result.context.text == direct_context.text
    assert len(result.context.chunks) == len(direct_context.chunks)
    for gathered, direct in zip(result.context.chunks, direct_context.chunks, strict=True):
        assert gathered.chunk.chunk_id == direct.chunk.chunk_id


@pytest.mark.asyncio
async def test_usage_summed_across_sub_queries() -> None:
    """Usage sums reranker.last_usage across all sub-queries."""
    chunk_a = make_chunk("c-a")
    chunk_b = make_chunk("c-b")

    retriever = StubRetriever()
    retriever.results["Q1"] = [make_scored_chunk(chunk_a)]
    retriever.results["Q2"] = [make_scored_chunk(chunk_b)]

    reranker = StubReranker()
    reranker.usages = [
        Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
        Usage(input_tokens=15, output_tokens=7, cost_usd=0.0015),
    ]

    plan = Plan(kind=PlanKind.MULTI_HOP, sub_queries=("Q1", "Q2"))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=1,
        max_context_tokens=1000,
        count_tokens=count_tokens,
    )

    expected = Usage(input_tokens=25, output_tokens=12, cost_usd=0.0025)
    assert result.usage == expected


@pytest.mark.asyncio
async def test_markers_are_global_one_to_n() -> None:
    """Context markers are 1..n in merged order."""
    chunk_a = make_chunk("c-a")
    chunk_b = make_chunk("c-b")
    chunk_c = make_chunk("c-c")

    retriever = StubRetriever()
    retriever.results["Q1"] = [
        make_scored_chunk(chunk_a),
        make_scored_chunk(chunk_b),
    ]
    retriever.results["Q2"] = [
        make_scored_chunk(chunk_c),
    ]

    reranker = StubReranker()
    reranker.usages = [Usage.zero(), Usage.zero()]

    plan = Plan(kind=PlanKind.MULTI_HOP, sub_queries=("Q1", "Q2"))

    result = await gather(
        retriever,
        reranker,
        plan,
        mode=RetrievalMode.HYBRID,
        candidate_pool=10,
        top_k=2,
        max_context_tokens=1000,
        count_tokens=count_tokens,
    )

    # Verify the context text has excerpt ids 1, 2, 3
    assert "<excerpt id=1 " in result.context.text
    assert "<excerpt id=2 " in result.context.text
    assert "<excerpt id=3 " in result.context.text

    # Verify chunks are in order
    assert result.context.chunks[0].chunk.chunk_id == "c-a"
    assert result.context.chunks[1].chunk.chunk_id == "c-b"
    assert result.context.chunks[2].chunk.chunk_id == "c-c"
