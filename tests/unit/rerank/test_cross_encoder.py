"""Tests for CrossEncoderReranker."""

from __future__ import annotations

import sys

import pytest

from agentic_rag.providers.base import Usage
from agentic_rag.rerank.base import Reranker
from agentic_rag.rerank.cross_encoder import CrossEncoderReranker
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    text: str = "The organization manages system accounts.",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=f"{section_id} SECTION",
        heading=f"{section_id} SECTION",
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_scored(
    chunk: ChunkRecord,
    score: float = 0.5,
    rank: int = 1,
    source_scores: dict[str, float] | None = None,
) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk=chunk,
        score=score,
        rank=rank,
        source_scores=source_scores or {"bm25": score},
    )


class FakeCrossEncoder:
    """Fake CrossEncoder model for testing."""

    def __init__(self) -> None:
        self.predict_calls: list[list[tuple[str, str]]] = []
        # Scores can be overridden per test
        self.scores: list[float] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Fake predict that records calls and returns scripted scores."""
        self.predict_calls.append(pairs)
        if self.scores is not None:
            return self.scores
        # Default: return higher scores for lower indices (c1 > c2 > c3 ...)
        return [1.0 - (0.1 * i) for i in range(len(pairs))]


async def test_reorders_by_descending_score():
    """Reorders candidates by score descending, ranks reassigned 1..n."""
    c1 = make_scored(make_chunk("c1"), score=0.8)
    c2 = make_scored(make_chunk("c2"), score=0.7)
    c3 = make_scored(make_chunk("c3"), score=0.6)
    candidates = [c1, c2, c3]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.6, 0.8, 0.7]  # c1=0.6, c2=0.8, c3=0.7
    reranker._model = fake_model

    result = await reranker.rerank("test query", candidates, top_k=3)

    assert len(result) == 3
    # c2 (0.8), c3 (0.7), c1 (0.6)
    assert result[0].chunk.chunk_id == "c2"
    assert result[1].chunk.chunk_id == "c3"
    assert result[2].chunk.chunk_id == "c1"
    assert [r.rank for r in result] == [1, 2, 3]


async def test_preserves_score_and_source_scores():
    """Preserves original score and source_scores from input."""
    c1 = make_scored(
        make_chunk("c1"),
        score=0.8,
        source_scores={"bm25": 0.8, "dense": 0.7},
    )
    c2 = make_scored(make_chunk("c2"), score=0.6, source_scores={"bm25": 0.6})
    candidates = [c1, c2]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.6, 0.8]  # c1=0.6, c2=0.8 (reordered)
    reranker._model = fake_model

    result = await reranker.rerank("query", candidates, top_k=2)

    # Result order is c2, c1 (by cross-encoder score)
    assert result[0].chunk.chunk_id == "c2"
    assert result[0].score == 0.6  # c2's original score
    assert result[0].source_scores == {"bm25": 0.6}
    assert result[1].chunk.chunk_id == "c1"
    assert result[1].score == 0.8  # c1's original score
    assert result[1].source_scores == {"bm25": 0.8, "dense": 0.7}


async def test_top_k_cuts():
    """top_k cuts candidates after reordering."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    c3 = make_scored(make_chunk("c3"))
    c4 = make_scored(make_chunk("c4"))
    c5 = make_scored(make_chunk("c5"))
    candidates = [c1, c2, c3, c4, c5]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.5, 0.4, 0.3, 0.2, 0.1]
    reranker._model = fake_model

    result = await reranker.rerank("query", candidates, top_k=3)

    assert len(result) == 3
    assert [r.chunk.chunk_id for r in result] == ["c1", "c2", "c3"]
    assert [r.rank for r in result] == [1, 2, 3]


async def test_top_k_greater_than_candidates():
    """top_k > len(candidates) returns all candidates."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.5, 0.4]
    reranker._model = fake_model

    result = await reranker.rerank("query", candidates, top_k=100)

    assert len(result) == 2
    assert [r.rank for r in result] == [1, 2]


async def test_ties_preserve_input_order():
    """Equal scores preserve input order (stable sort)."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    c3 = make_scored(make_chunk("c3"))
    candidates = [c1, c2, c3]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.5, 0.5, 0.5]  # All tied
    reranker._model = fake_model

    result = await reranker.rerank("query", candidates, top_k=3)

    # Input order preserved on ties
    assert [r.chunk.chunk_id for r in result] == ["c1", "c2", "c3"]
    assert [r.rank for r in result] == [1, 2, 3]


async def test_empty_candidates():
    """Empty candidates returns [] and last_usage == Usage.zero()."""
    reranker = CrossEncoderReranker()
    result = await reranker.rerank("query", [], top_k=10)

    assert result == []
    assert reranker.last_usage == Usage.zero()


async def test_last_usage_is_zero_after_rerank():
    """last_usage is Usage.zero() after a successful rerank."""
    c1 = make_scored(make_chunk("c1"))
    candidates = [c1]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.5]
    reranker._model = fake_model

    await reranker.rerank("query", candidates, top_k=1)

    assert reranker.last_usage == Usage.zero()


async def test_missing_dependency_error():
    """Importing sentence_transformers fails raises ImportError with helpful message."""
    # Temporarily hide sentence_transformers
    old_modules = {}
    try:
        old_modules["sentence_transformers"] = sys.modules.pop("sentence_transformers", None)
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]

        reranker = CrossEncoderReranker()
        c1 = make_scored(make_chunk("c1"))

        with pytest.raises(ImportError) as exc_info:
            await reranker.rerank("query", [c1], top_k=1)

        assert "sentence-transformers" in str(exc_info.value)
        assert "rerank-local" in str(exc_info.value)

    finally:
        # Restore
        if (
            "sentence_transformers" in old_modules
            and old_modules["sentence_transformers"] is not None
        ):
            sys.modules["sentence_transformers"] = old_modules["sentence_transformers"]
        elif "sentence_transformers" in sys.modules:
            del sys.modules["sentence_transformers"]


async def test_is_reranker_protocol():
    """CrossEncoderReranker satisfies Reranker protocol."""
    reranker = CrossEncoderReranker()
    assert isinstance(reranker, Reranker)


async def test_predict_called_with_pairs_in_candidate_order():
    """predict is called with (query, chunk_text) pairs in candidate order."""
    c1 = make_scored(make_chunk("c1", text="Text for c1"))
    c2 = make_scored(make_chunk("c2", text="Text for c2"))
    c3 = make_scored(make_chunk("c3", text="Text for c3"))
    candidates = [c1, c2, c3]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.5, 0.6, 0.4]
    reranker._model = fake_model

    await reranker.rerank("test query", candidates, top_k=3)

    assert len(fake_model.predict_calls) == 1
    pairs = fake_model.predict_calls[0]
    assert len(pairs) == 3
    # Pairs are in candidate order: (query, chunk_text)
    assert pairs[0] == ("test query", "Text for c1")
    assert pairs[1] == ("test query", "Text for c2")
    assert pairs[2] == ("test query", "Text for c3")


async def test_default_model_name():
    """Default model name is BAAI/bge-reranker-base."""
    reranker = CrossEncoderReranker()
    assert reranker._model_name == "BAAI/bge-reranker-base"


async def test_custom_model_name():
    """Custom model name is stored and used."""
    reranker = CrossEncoderReranker(model="custom-model")
    assert reranker._model_name == "custom-model"


async def test_name_attribute():
    """name attribute is 'cross-encoder'."""
    reranker = CrossEncoderReranker()
    assert reranker.name == "cross-encoder"


async def test_multiple_candidates_reordered_by_score():
    """Multiple candidates reordered by descending score."""
    candidates = [
        make_scored(make_chunk("a")),
        make_scored(make_chunk("b")),
        make_scored(make_chunk("c")),
        make_scored(make_chunk("d")),
    ]

    reranker = CrossEncoderReranker()
    fake_model = FakeCrossEncoder()
    fake_model.scores = [0.3, 0.7, 0.9, 0.5]  # c, b, a, d order
    reranker._model = fake_model

    result = await reranker.rerank("query", candidates, top_k=4)

    # Expected order by descending score: c(0.9), b(0.7), d(0.5), a(0.3)
    assert [r.chunk.chunk_id for r in result] == ["c", "b", "d", "a"]
    assert [r.rank for r in result] == [1, 2, 3, 4]


async def test_initial_last_usage_is_zero():
    """Initial last_usage is Usage.zero()."""
    reranker = CrossEncoderReranker()
    assert reranker.last_usage == Usage.zero()
