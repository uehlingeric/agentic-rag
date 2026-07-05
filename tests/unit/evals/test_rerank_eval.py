"""Tests for reranker evaluation harness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from agentic_rag.evals.rerank import run_rerank_eval
from agentic_rag.evals.retrieval import Citation, GoldenExample
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc1",
    section_id: str = "S1",
    section_ids: list[str] | None = None,
    text: str = "chunk text",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=section_ids if section_ids is not None else [section_id],
        section_path="section",
        heading="heading",
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_scored(chunk: ChunkRecord, rank: int, score: float = 1.0) -> ScoredChunk:
    """Helper to create a ScoredChunk."""
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def make_golden(
    qid: str,
    question: str,
    citations: list[Citation],
    *,
    type_: str = "lookup",
) -> GoldenExample:
    """Helper to create a GoldenExample."""
    return GoldenExample(
        id=qid,
        question=question,
        reference_answer="A",
        source_citations=citations,
        difficulty="easy",
        type=type_,
    )


class StubRetriever:
    """Stub retriever that tracks calls and returns preset pools per (query, mode)."""

    def __init__(self, pools: dict[tuple[str, str], list[ScoredChunk]]) -> None:
        self.pools = pools
        self.retrieve_calls: list[tuple[str, str, int]] = []

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        self.retrieve_calls.append((query, mode.value, top_k))
        return self.pools.get((query, mode.value), [])


class StubReranker:
    """Stub reranker applying a deterministic reorder (default: reverse the cut)."""

    def __init__(
        self,
        name: str = "stub",
        reorder_fn: Callable[[Sequence[ScoredChunk], int], list[ScoredChunk]] | None = None,
        usage_per_call: Usage | None = None,
    ) -> None:
        self.name = name
        self.last_usage = Usage.zero()
        self.reorder_fn = reorder_fn or (
            lambda candidates, top_k: list(reversed(candidates[:top_k]))
        )
        self.usage_per_call = usage_per_call or Usage.zero()
        self.rerank_calls: list[tuple[str, int, int]] = []

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        self.rerank_calls.append((query, len(candidates), top_k))
        self.last_usage = self.usage_per_call
        reordered = self.reorder_fn(candidates, top_k)
        return [replace(c, rank=i) for i, c in enumerate(reordered, start=1)]


async def test_reranker_improves_metrics() -> None:
    """Reranker that moves the covering chunk from rank 3 to rank 1 improves NDCG and MRR.

    Baseline (retrieval order): covering chunk at rank 3 -> MRR = 1/3,
    NDCG@10 = (1/log2(4)) / (1/log2(2)) = 0.5.
    Reranked (reversed): covering chunk at rank 1 -> MRR = 1.0, NDCG@10 = 1.0.
    """
    pool = [
        make_scored(make_chunk("c1", doc_id="doc2"), rank=1),
        make_scored(make_chunk("c2", doc_id="doc3"), rank=2),
        make_scored(make_chunk("c3", doc_id="doc1", section_ids=["S1"]), rank=3),
    ]
    golden = [make_golden("q1", "Test?", [Citation("doc1", "S1")])]
    retriever = StubRetriever({("Test?", "hybrid"): pool})
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=3,
        top_k=3,
    )

    assert report.n_answerable == 1
    assert report.n_skipped_unanswerable == 0
    assert len(report.modes) == 2

    baseline_mode, reranked_mode = report.modes
    assert baseline_mode.mode == "hybrid"
    assert reranked_mode.mode == "hybrid+stub"

    assert baseline_mode.metrics["mrr"] == pytest.approx(1.0 / 3.0)
    assert baseline_mode.metrics["ndcg@10"] == pytest.approx(0.5)
    assert reranked_mode.metrics["mrr"] == pytest.approx(1.0)
    assert reranked_mode.metrics["ndcg@10"] == pytest.approx(1.0)


async def test_retriever_called_once_per_mode_question_with_pool_size() -> None:
    """Retriever called exactly once per (mode, question) with top_k = pool."""
    golden = [
        make_golden("q1", "Q1?", [Citation("doc1", "S1")]),
        make_golden("q2", "Q2?", [Citation("doc2", "S2")]),
    ]
    pools = {
        ("Q1?", "hybrid"): [make_scored(make_chunk(f"a{i}"), rank=i) for i in range(1, 11)],
        ("Q2?", "hybrid"): [make_scored(make_chunk(f"b{i}"), rank=i) for i in range(1, 11)],
    }
    retriever = StubRetriever(pools)
    reranker = StubReranker()

    await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=10,
        top_k=5,
    )

    assert retriever.retrieve_calls == [("Q1?", "hybrid", 10), ("Q2?", "hybrid", 10)]


async def test_reranker_called_with_full_pool_and_top_k() -> None:
    """Reranker receives the full pool and the metric top_k (not the pool size)."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk(f"c{i}"), rank=i) for i in range(1, 31)]
    retriever = StubRetriever({("Q?", "bm25"): pool})
    reranker = StubReranker()

    await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["bm25"],
        pool=30,
        top_k=10,
    )

    assert reranker.rerank_calls == [("Q?", 30, 10)]


async def test_mode_labels_with_reranker_name() -> None:
    """ModeReport labels are mode and mode+reranker.name."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    retriever = StubRetriever({("Q?", "dense"): pool})
    reranker = StubReranker(name="my-reranker")

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["dense"],
        pool=1,
        top_k=1,
    )

    assert report.modes[0].mode == "dense"
    assert report.modes[1].mode == "dense+my-reranker"


async def test_multiple_modes_ordered() -> None:
    """Multiple modes produce baseline+reranked pairs in input order."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    pools = {("Q?", m): pool for m in ("bm25", "dense", "hybrid")}
    retriever = StubRetriever(pools)
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["bm25", "dense", "hybrid"],
        pool=1,
        top_k=1,
    )

    assert [mr.mode for mr in report.modes] == [
        "bm25",
        "bm25+stub",
        "dense",
        "dense+stub",
        "hybrid",
        "hybrid+stub",
    ]


async def test_unanswerable_skipped() -> None:
    """Unanswerable examples are not retrieved and not reranked."""
    golden = [
        make_golden("q1", "Q1?", [Citation("doc1", "S1")]),
        make_golden("q2", "Unanswerable?", [], type_="unanswerable"),
    ]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    retriever = StubRetriever({("Q1?", "hybrid"): pool})
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=1,
        top_k=1,
    )

    assert report.n_answerable == 1
    assert report.n_skipped_unanswerable == 1
    assert len(retriever.retrieve_calls) == 1
    assert len(reranker.rerank_calls) == 1


async def test_usage_totals_summed() -> None:
    """Usage totals summed across all rerank calls."""
    golden = [
        make_golden("q1", "Q1?", [Citation("doc1", "S1")]),
        make_golden("q2", "Q2?", [Citation("doc2", "S2")]),
    ]
    pools = {
        ("Q1?", "hybrid"): [make_scored(make_chunk("c1"), rank=1)],
        ("Q2?", "hybrid"): [make_scored(make_chunk("c2"), rank=1)],
    }
    retriever = StubRetriever(pools)
    reranker = StubReranker(usage_per_call=Usage(input_tokens=10, output_tokens=5))

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=1,
        top_k=1,
    )

    assert report.config["rerank_total_input_tokens"] == 20
    assert report.config["rerank_total_output_tokens"] == 10


async def test_cost_usd_summed_and_handles_none() -> None:
    """Cost USD summed; None cost treated as 0.0."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    retriever = StubRetriever({("Q?", "hybrid"): pool})
    reranker = StubReranker(usage_per_call=Usage(10, 5, cost_usd=None))

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=1,
        top_k=1,
    )

    assert report.config["rerank_total_cost_usd"] == 0.0


async def test_mean_seconds_per_query() -> None:
    """rerank_mean_seconds_per_query is total / call count and non-negative."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    retriever = StubRetriever({("Q?", "hybrid"): pool})
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=1,
        top_k=1,
    )

    mean_seconds = report.config["rerank_mean_seconds_per_query"]
    assert isinstance(mean_seconds, float)
    assert mean_seconds >= 0.0


async def test_config_preserved_and_augmented() -> None:
    """Passed-in config keys preserved alongside the augmented rerank keys."""
    golden = [make_golden("q1", "Q?", [Citation("doc1", "S1")])]
    pool = [make_scored(make_chunk("c1"), rank=1)]
    retriever = StubRetriever({("Q?", "hybrid"): pool})
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=10,
        top_k=5,
        config={"my_key": "my_value", "number": 42},
    )

    assert report.config["my_key"] == "my_value"
    assert report.config["number"] == 42
    assert report.config["reranker"] == "stub"
    assert report.config["pool"] == 10
    assert report.config["top_k"] == 5
    assert "rerank_total_input_tokens" in report.config
    assert "rerank_total_output_tokens" in report.config
    assert "rerank_total_cost_usd" in report.config
    assert "rerank_mean_seconds_per_query" in report.config


async def test_per_query_metrics_recorded() -> None:
    """per_query has one entry per answerable question with all six metric keys."""
    golden = [
        make_golden("q1", "Q1?", [Citation("doc1", "S1")]),
        make_golden("q2", "Q2?", [Citation("doc2", "S2")]),
    ]
    pools = {
        ("Q1?", "hybrid"): [
            make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1)
        ],
        ("Q2?", "hybrid"): [
            make_scored(make_chunk("c2", doc_id="doc2", section_ids=["S2"]), rank=1)
        ],
    }
    retriever = StubRetriever(pools)
    reranker = StubReranker()

    report = await run_rerank_eval(
        retriever,  # type: ignore[arg-type]
        reranker,
        golden,
        modes=["hybrid"],
        pool=1,
        top_k=1,
    )

    baseline_mode, reranked_mode = report.modes
    for mode_report in (baseline_mode, reranked_mode):
        assert set(mode_report.per_query) == {"q1", "q2"}
        for query_metrics in mode_report.per_query.values():
            assert set(query_metrics.keys()) == {
                "recall@5",
                "recall@10",
                "recall@20",
                "precision@5",
                "mrr",
                "ndcg@10",
            }
