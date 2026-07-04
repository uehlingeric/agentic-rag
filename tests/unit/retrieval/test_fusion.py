"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.base import ScoredChunk
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from tests.unit.retrieval.conftest import make_chunk


class TestReciprocalRankFusion:
    """Test reciprocal rank fusion math and ordering."""

    def test_two_rankings_with_shared_chunk(self) -> None:
        """Fuse two rankings with one shared chunk.

        BM25 ranking: [c1 (rank 1), c2 (rank 2), c3 (rank 3)]
        Dense ranking: [c1 (rank 1), c4 (rank 2), c5 (rank 3)]

        Shared chunk c1:
          score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787

        Disjoint c2 (rank 2 in bm25 only):
          score = 1/(60+2) = 1/62 ≈ 0.016129

        Disjoint c3 (rank 3 in bm25 only):
          score = 1/(60+3) = 1/63 ≈ 0.015873

        Disjoint c4 (rank 2 in dense only):
          score = 1/(60+2) = 1/62 ≈ 0.016129

        Disjoint c5 (rank 3 in dense only):
          score = 1/(60+3) = 1/63 ≈ 0.015873

        Order: c1 (0.032787), then c2 and c4 tie (0.016129 each),
        break tie by chunk_id: c2 < c4, then c3 and c5 tie (0.015873),
        break tie by chunk_id: c3 < c5.
        """
        c1 = ScoredChunk(chunk=make_chunk("c1"), score=1.0, rank=1, source_scores={})
        c2 = ScoredChunk(chunk=make_chunk("c2"), score=0.9, rank=2, source_scores={})
        c3 = ScoredChunk(chunk=make_chunk("c3"), score=0.8, rank=3, source_scores={})
        c4 = ScoredChunk(chunk=make_chunk("c4"), score=0.9, rank=2, source_scores={})
        c5 = ScoredChunk(chunk=make_chunk("c5"), score=0.8, rank=3, source_scores={})

        bm25_ranking = [c1, c2, c3]
        dense_ranking = [c1, c4, c5]

        result = reciprocal_rank_fusion({"bm25": bm25_ranking, "dense": dense_ranking}, k=60)

        assert len(result) == 5

        # Check shared chunk score
        assert result[0].chunk.chunk_id == "c1"
        assert result[0].score == pytest.approx(2 / 61, abs=1e-6)
        assert result[0].rank == 1

        # Check tie-breaking: c2 and c4 both score 1/62, c2 comes first (c2 < c4)
        assert result[1].chunk.chunk_id == "c2"
        assert result[1].score == pytest.approx(1 / 62, abs=1e-6)
        assert result[2].chunk.chunk_id == "c4"
        assert result[2].score == pytest.approx(1 / 62, abs=1e-6)

        # c3 and c5 both score 1/63, c3 comes first
        assert result[3].chunk.chunk_id == "c3"
        assert result[3].score == pytest.approx(1 / 63, abs=1e-6)
        assert result[4].chunk.chunk_id == "c5"
        assert result[4].score == pytest.approx(1 / 63, abs=1e-6)

    def test_rank_reassignment_contiguous(self) -> None:
        """Ranks reassigned 1..n contiguously."""
        chunks = [
            ScoredChunk(chunk=make_chunk(f"c{i}"), score=float(10 - i), rank=i) for i in range(1, 4)
        ]

        result = reciprocal_rank_fusion({"mode1": chunks}, k=60)

        assert len(result) == 3
        for idx, r in enumerate(result, start=1):
            assert r.rank == idx

    def test_source_scores_merged(self) -> None:
        """source_scores from all rankings merged for shared chunks."""
        c1_bm25 = ScoredChunk(
            chunk=make_chunk("shared"),
            score=0.9,
            rank=1,
            source_scores={"bm25": 10.5},
        )
        c1_dense = ScoredChunk(
            chunk=make_chunk("shared"),
            score=0.8,
            rank=1,
            source_scores={"dense": 0.95},
        )

        result = reciprocal_rank_fusion({"bm25": [c1_bm25], "dense": [c1_dense]}, k=60)

        assert len(result) == 1
        assert result[0].source_scores == {"bm25": 10.5, "dense": 0.95}

    def test_k_parameter_affects_scores(self) -> None:
        """Different k values change RRF scores as expected."""
        chunk = ScoredChunk(chunk=make_chunk("c1"), score=1.0, rank=1)

        result_k60 = reciprocal_rank_fusion({"mode1": [chunk]}, k=60)
        result_k100 = reciprocal_rank_fusion({"mode1": [chunk]}, k=100)

        # k=60: 1/61, k=100: 1/101
        assert result_k60[0].score == pytest.approx(1 / 61, abs=1e-6)
        assert result_k100[0].score == pytest.approx(1 / 101, abs=1e-6)
        assert result_k60[0].score > result_k100[0].score

    def test_empty_mapping_returns_empty_list(self) -> None:
        """Empty rankings mapping returns empty list."""
        result = reciprocal_rank_fusion({}, k=60)
        assert result == []

    def test_mapping_of_empty_lists_returns_empty_list(self) -> None:
        """Mapping with all empty lists returns empty list."""
        result = reciprocal_rank_fusion({"bm25": [], "dense": []}, k=60)
        assert result == []

    def test_determinism_same_input_produces_identical_output(self) -> None:
        """Same input produces identical output on repeated calls."""
        chunks = [
            ScoredChunk(chunk=make_chunk("c1"), score=1.0, rank=1),
            ScoredChunk(chunk=make_chunk("c2"), score=0.8, rank=2),
        ]
        rankings = {"mode1": chunks}

        result1 = reciprocal_rank_fusion(rankings, k=60)
        result2 = reciprocal_rank_fusion(rankings, k=60)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank

    def test_single_mode_ranking(self) -> None:
        """Single mode ranking is fused correctly."""
        chunks = [
            ScoredChunk(chunk=make_chunk("c1"), score=1.0, rank=1),
            ScoredChunk(chunk=make_chunk("c2"), score=0.5, rank=2),
        ]

        result = reciprocal_rank_fusion({"bm25": chunks}, k=60)

        assert len(result) == 2
        assert result[0].chunk.chunk_id == "c1"
        assert result[0].score == pytest.approx(1 / 61, abs=1e-6)
        assert result[1].chunk.chunk_id == "c2"
        assert result[1].score == pytest.approx(1 / 62, abs=1e-6)

    def test_many_modes_same_chunk(self) -> None:
        """Same chunk appearing in many modes accumulates scores."""
        chunk = ScoredChunk(chunk=make_chunk("shared"), score=1.0, rank=1)

        result = reciprocal_rank_fusion(
            {
                "mode1": [chunk],
                "mode2": [chunk],
                "mode3": [chunk],
            },
            k=60,
        )

        assert len(result) == 1
        # 3 appearances at rank 1: 3 * (1/61)
        assert result[0].score == pytest.approx(3 / 61, abs=1e-6)

    def test_tie_breaking_deterministic(self) -> None:
        """Tie-breaking by chunk_id is deterministic.

        b_chunk at rank 1 in mode1: 1/61 ≈ 0.01639
        a_chunk at rank 2 in mode1: 1/62 ≈ 0.01613
        c_chunk at rank 1 in mode2: 1/61 ≈ 0.01639

        Sorted by score DESC, then chunk_id ASC:
        1. b_chunk (0.01639) - comes before c_chunk alphabetically
        2. c_chunk (0.01639)
        3. a_chunk (0.01613)
        """
        chunks_a = [
            ScoredChunk(chunk=make_chunk("b_chunk"), score=1.0, rank=1),
            ScoredChunk(chunk=make_chunk("a_chunk"), score=1.0, rank=2),
        ]
        chunks_b = [
            ScoredChunk(chunk=make_chunk("c_chunk"), score=1.0, rank=1),
        ]

        result = reciprocal_rank_fusion({"mode1": chunks_a, "mode2": chunks_b}, k=60)

        # Order by score DESC then chunk_id ASC
        assert result[0].chunk.chunk_id == "b_chunk"
        assert result[0].score == pytest.approx(1 / 61, abs=1e-6)
        assert result[1].chunk.chunk_id == "c_chunk"
        assert result[1].score == pytest.approx(1 / 61, abs=1e-6)
        assert result[2].chunk.chunk_id == "a_chunk"
        assert result[2].score == pytest.approx(1 / 62, abs=1e-6)

    def test_large_rank_differences(self) -> None:
        """Large rank differences in source rankings correctly affect fusion."""
        c1_rank1 = ScoredChunk(chunk=make_chunk("c1"), score=1.0, rank=1)
        c1_rank100 = ScoredChunk(chunk=make_chunk("c1"), score=0.01, rank=100)

        # When fused: 1/61 + 1/160 ≈ 0.01902
        result = reciprocal_rank_fusion({"mode1": [c1_rank1], "mode2": [c1_rank100]}, k=60)

        assert len(result) == 1
        expected = 1 / 61 + 1 / 160
        assert result[0].score == pytest.approx(expected, abs=1e-6)
