"""Tests for BM25 sparse retrieval index."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.retrieval.base import ChunkRecord
from agentic_rag.retrieval.bm25 import BM25Index


@pytest.fixture()
def bm25_db(tmp_path: Path, tiny_corpus: list[ChunkRecord]) -> Path:
    """Build a BM25 index from tiny_corpus and return the db path."""
    db_path = tmp_path / "test_bm25.db"
    BM25Index.build(tiny_corpus, db_path)
    return db_path


class TestBM25BuildAndOpen:
    """Test index creation and reopening."""

    def test_build_creates_index(self, tmp_path: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that build creates an index file."""
        db_path = tmp_path / "new_index.db"
        index = BM25Index.build(tiny_corpus, db_path)
        assert db_path.exists()
        assert index.size == 5
        index.close()

    def test_reopen_index(self, tmp_path: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that built index can be reopened."""
        db_path = tmp_path / "index.db"
        index1 = BM25Index.build(tiny_corpus, db_path)
        index1.close()

        # Reopen the same index
        index2 = BM25Index(db_path)
        assert index2.size == 5
        index2.close()

    def test_open_missing_path_raises(self, tmp_path: Path) -> None:
        """Test that opening a missing path raises FileNotFoundError."""
        db_path = tmp_path / "nonexistent.db"
        with pytest.raises(
            FileNotFoundError,
            match="BM25 index not found at",
        ):
            BM25Index(db_path)

    def test_build_overwrites_existing(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that build overwrites existing index."""
        db_path = tmp_path / "index.db"

        # Build first time
        index1 = BM25Index.build(tiny_corpus, db_path)
        index1.close()

        # Build again — should succeed and not raise
        index2 = BM25Index.build(tiny_corpus, db_path)
        assert index2.size == 5
        index2.close()

    def test_build_creates_parent_directories(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that build creates parent directories."""
        db_path = tmp_path / "deeply" / "nested" / "path" / "index.db"
        index = BM25Index.build(tiny_corpus, db_path)
        assert db_path.exists()
        index.close()


class TestBM25Search:
    """Test search functionality."""

    def test_search_account_management(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that 'account management' ranks c-access first."""
        index = BM25Index(bm25_db)
        results = index.search("account management", top_k=10)
        index.close()

        assert len(results) > 0
        assert results[0].chunk.chunk_id == "c-access"

    def test_search_control_id_ac2(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that 'AC-2' ranks c-access first via phrase matching."""
        index = BM25Index(bm25_db)
        results = index.search("AC-2", top_k=10)
        index.close()

        assert len(results) > 0
        assert results[0].chunk.chunk_id == "c-access"

    def test_search_empty_query_returns_empty(self, bm25_db: Path) -> None:
        """Test that empty query returns empty list."""
        index = BM25Index(bm25_db)
        results = index.search("", top_k=10)
        index.close()

        assert results == []

    def test_search_punctuation_only_returns_empty(self, bm25_db: Path) -> None:
        """Test that query with only punctuation returns empty list."""
        index = BM25Index(bm25_db)
        results = index.search("!@#$%^&*()", top_k=10)
        index.close()

        assert results == []

    def test_search_no_match_returns_empty(self, bm25_db: Path) -> None:
        """Test that query with no matches returns empty list."""
        index = BM25Index(bm25_db)
        results = index.search("xyzabc123nonexistent", top_k=10)
        index.close()

        assert results == []

    def test_search_top_k_respected(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that top_k limit is respected."""
        index = BM25Index(bm25_db)

        # Search for broad terms that might match multiple chunks
        results = index.search("security", top_k=2)
        index.close()

        # Should have at most 2 results
        assert len(results) <= 2

    def test_search_fts5_injection_or(self, bm25_db: Path) -> None:
        """Test that FTS5 injection with OR operator is handled safely."""
        index = BM25Index(bm25_db)
        # This should not raise and should return safely
        results = index.search('text OR (NEAR "foo"', top_k=10)
        index.close()

        # Should return some results or empty, but not raise
        assert isinstance(results, list)

    def test_search_fts5_injection_unclosed_quote(self, bm25_db: Path) -> None:
        """Test that unclosed FTS5 quotes are handled safely."""
        index = BM25Index(bm25_db)
        # This should not raise and should return safely
        results = index.search('"unclosed', top_k=10)
        index.close()

        assert isinstance(results, list)

    def test_search_fts5_injection_column_syntax(self, bm25_db: Path) -> None:
        """Test that FTS5 column syntax is handled safely."""
        index = BM25Index(bm25_db)
        # This should not raise and should return safely
        results = index.search("col:val*", top_k=10)
        index.close()

        assert isinstance(results, list)

    def test_search_determinism_same_query_twice(
        self, bm25_db: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that same query returns identical results."""
        index = BM25Index(bm25_db)
        results1 = index.search("account", top_k=10)
        results2 = index.search("account", top_k=10)
        index.close()

        # Compare results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank

    def test_search_determinism_rebuild(
        self,
        tmp_path: Path,
        tiny_corpus: list[ChunkRecord],
    ) -> None:
        """Test that rebuild into second path gives identical results."""
        db_path1 = tmp_path / "index1.db"
        db_path2 = tmp_path / "index2.db"

        # Build two indexes with same corpus
        BM25Index.build(tiny_corpus, db_path1)
        BM25Index.build(tiny_corpus, db_path2)

        index1 = BM25Index(db_path1)
        index2 = BM25Index(db_path2)

        results1 = index1.search("account", top_k=10)
        results2 = index2.search("account", top_k=10)

        index1.close()
        index2.close()

        # Compare results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank


class TestBM25Scores:
    """Test score calculation and normalization."""

    def test_top_result_score_is_one(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that top result has score == 1.0."""
        index = BM25Index(bm25_db)
        results = index.search("account management", top_k=10)
        index.close()

        assert len(results) > 0
        assert results[0].score == 1.0

    def test_bm25_score_in_source_scores(
        self, bm25_db: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that source_scores contains 'bm25' key."""
        index = BM25Index(bm25_db)
        results = index.search("account", top_k=10)
        index.close()

        assert len(results) > 0
        for result in results:
            assert "bm25" in result.source_scores
            assert isinstance(result.source_scores["bm25"], float)

    def test_bm25_scores_positive(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that bm25 raw scores are positive."""
        index = BM25Index(bm25_db)
        results = index.search("account", top_k=10)
        index.close()

        assert len(results) > 0
        for result in results:
            assert result.source_scores["bm25"] > 0

    def test_bm25_scores_non_increasing(
        self, bm25_db: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that bm25 raw scores are non-increasing with rank."""
        index = BM25Index(bm25_db)
        results = index.search("account", top_k=10)
        index.close()

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].source_scores["bm25"] >= results[i + 1].source_scores["bm25"]

    def test_normalized_scores_non_increasing(
        self, bm25_db: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Test that normalized scores are non-increasing with rank."""
        index = BM25Index(bm25_db)
        results = index.search("account", top_k=10)
        index.close()

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestBM25Rank:
    """Test rank field."""

    def test_rank_is_one_based(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that rank is 1-based."""
        index = BM25Index(bm25_db)
        results = index.search("account", top_k=10)
        index.close()

        for idx, result in enumerate(results, start=1):
            assert result.rank == idx

    def test_rank_sequential(self, bm25_db: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that ranks are sequential."""
        index = BM25Index(bm25_db)
        results = index.search("security", top_k=10)
        index.close()

        if results:
            for idx, result in enumerate(results, start=1):
                assert result.rank == idx


class TestBM25Size:
    """Test size property."""

    def test_size_property(self, tmp_path: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Test that size property returns correct count."""
        db_path = tmp_path / "index.db"
        index = BM25Index.build(tiny_corpus, db_path)
        assert index.size == 5
        index.close()

    def test_size_empty_index(self, tmp_path: Path) -> None:
        """Test size property on empty index."""
        db_path = tmp_path / "empty.db"
        index = BM25Index.build([], db_path)
        assert index.size == 0
        index.close()
