"""Tests for unified Retriever."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.providers.base import EmbeddingResult, Usage
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode
from agentic_rag.retrieval.bm25 import BM25Index
from agentic_rag.retrieval.dense import DenseIndex
from agentic_rag.retrieval.embed import EmbeddingMatrix
from agentic_rag.retrieval.retriever import Retriever


class FakeEmbedder:
    """Deterministic embedder for testing."""

    name = "fake"

    def __init__(self, vectors_by_text: dict[str, list[float]] | None = None) -> None:
        """Initialize with optional predefined vectors for specific texts.

        Args:
            vectors_by_text: Mapping of text to vector. If text not found,
                             defaults to zero vector.
        """
        self.vectors_by_text = vectors_by_text or {}

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult:
        """Return predefined or default vectors."""
        vectors = []
        for text in texts:
            if text in self.vectors_by_text:
                vectors.append(self.vectors_by_text[text])
            else:
                # Default: zero vector (or could be random)
                vectors.append([0.0] * 8)

        return EmbeddingResult(
            vectors=vectors,
            model=model or "fake",
            dimensions=8,
            usage=Usage(input_tokens=len(texts), output_tokens=0),
        )


class TestRetrieverModeDispatch:
    """Test retrieval mode dispatch to underlying indexes."""

    @pytest.mark.asyncio
    async def test_bm25_mode_returns_bm25_results(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """BM25 mode returns results from BM25 index only."""
        # Build BM25 index
        bm25_path = tmp_path / "bm25.db"
        bm25 = BM25Index.build(tiny_corpus, bm25_path)

        # Build dense index with random vectors
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        index_dir = tmp_path / "dense"
        dense = DenseIndex.build(matrix, tiny_corpus, index_dir)

        # Create retriever
        embedder = FakeEmbedder()
        retriever = Retriever(bm25, dense, embedder)

        # Search in BM25 mode
        results = await retriever.retrieve("account", mode=RetrievalMode.BM25, top_k=10)

        # Verify results come from BM25
        assert len(results) > 0
        # BM25 should rank "account" related chunks first
        assert results[0].chunk.chunk_id == "c-access"

    @pytest.mark.asyncio
    async def test_dense_mode_returns_dense_results(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """DENSE mode returns results from dense index only."""
        # Build BM25 (unused in this test)
        bm25_path = tmp_path / "bm25.db"
        bm25 = BM25Index.build(tiny_corpus, bm25_path)

        # Build dense with hand-set vectors for known ordering
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-access
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-audit
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-risk
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # c-crypto
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # c-training
            ],
            dtype=np.float32,
        )
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        index_dir = tmp_path / "dense"
        dense = DenseIndex.build(matrix, tiny_corpus, index_dir)

        # Embedder returns vector similar to first chunk
        embedder = FakeEmbedder(
            {
                "account": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        retriever = Retriever(bm25, dense, embedder)

        # Search in DENSE mode
        results = await retriever.retrieve("account", mode=RetrievalMode.DENSE, top_k=10)

        # Should get c-access first
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "c-access"

    @pytest.mark.asyncio
    async def test_hybrid_mode_fuses_results(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """HYBRID mode fuses BM25 and dense results."""
        # Build BM25
        bm25_path = tmp_path / "bm25.db"
        bm25 = BM25Index.build(tiny_corpus, bm25_path)

        # Build dense with orthogonal vectors
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-access
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-audit
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-risk
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # c-crypto
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # c-training
            ],
            dtype=np.float32,
        )
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        index_dir = tmp_path / "dense"
        dense = DenseIndex.build(matrix, tiny_corpus, index_dir)

        # Embedder returns vector for dense ranking
        embedder = FakeEmbedder(
            {
                "account": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        retriever = Retriever(bm25, dense, embedder)

        # Search in HYBRID mode
        results = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)

        # Should return at most 5 results
        assert len(results) <= 5
        # Chunk ranked top by both modes should be first
        assert results[0].chunk.chunk_id == "c-access"

    @pytest.mark.asyncio
    async def test_hybrid_result_length(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Hybrid results respect top_k limit."""
        bm25_path = tmp_path / "bm25.db"
        bm25 = BM25Index.build(tiny_corpus, bm25_path)

        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        index_dir = tmp_path / "dense"
        dense = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()
        retriever = Retriever(bm25, dense, embedder)

        results = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_hybrid_rank_contiguity(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Hybrid results have contiguous ranks 1..n after cut."""
        bm25_path = tmp_path / "bm25.db"
        bm25 = BM25Index.build(tiny_corpus, bm25_path)

        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        index_dir = tmp_path / "dense"
        dense = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()
        retriever = Retriever(bm25, dense, embedder)

        results = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)

        # Ranks should be 1, 2, 3, ..., len(results)
        for idx, result in enumerate(results, start=1):
            assert result.rank == idx


class TestRetrieverLoad:
    """Test Retriever.load() from persisted indexes."""

    @pytest.mark.asyncio
    async def test_load_and_retrieve(self, tmp_path: Path, tiny_corpus: list[ChunkRecord]) -> None:
        """Retriever.load() works and retrieve works in all modes."""
        # Build both indexes in same directory
        bm25_path = tmp_path / "bm25.db"
        BM25Index.build(tiny_corpus, bm25_path)

        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        DenseIndex.build(matrix, tiny_corpus, tmp_path)

        # Load retriever
        embedder = FakeEmbedder()
        retriever = Retriever.load(tmp_path, embedder)

        # Test all modes
        bm25_results = await retriever.retrieve("account", mode=RetrievalMode.BM25, top_k=5)
        assert len(bm25_results) > 0

        dense_results = await retriever.retrieve("account", mode=RetrievalMode.DENSE, top_k=5)
        assert len(dense_results) > 0

        hybrid_results = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)
        assert len(hybrid_results) > 0

    @pytest.mark.asyncio
    async def test_load_missing_bm25_raises(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Retriever.load() raises FileNotFoundError if BM25 missing."""
        # Build only dense index
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        DenseIndex.build(matrix, tiny_corpus, tmp_path)

        embedder = FakeEmbedder()
        with pytest.raises(FileNotFoundError):
            Retriever.load(tmp_path, embedder)

    @pytest.mark.asyncio
    async def test_load_missing_dense_raises(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Retriever.load() raises FileNotFoundError if dense missing."""
        # Build only BM25 index
        bm25_path = tmp_path / "bm25.db"
        BM25Index.build(tiny_corpus, bm25_path)

        embedder = FakeEmbedder()
        with pytest.raises(FileNotFoundError):
            Retriever.load(tmp_path, embedder)

    @pytest.mark.asyncio
    async def test_load_with_custom_rrf_k(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Retriever.load() respects custom rrf_k and candidate_pool."""
        bm25_path = tmp_path / "bm25.db"
        BM25Index.build(tiny_corpus, bm25_path)

        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        DenseIndex.build(matrix, tiny_corpus, tmp_path)

        embedder = FakeEmbedder()
        retriever = Retriever.load(tmp_path, embedder, rrf_k=100, candidate_pool=30)

        # Just verify it loaded with custom params (integration check)
        results = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)
        assert len(results) > 0


class TestRetrieverDeterminism:
    """Test Retriever determinism."""

    @pytest.mark.asyncio
    async def test_same_query_same_mode_produces_same_results(
        self, tmp_path: Path, tiny_corpus: list[ChunkRecord]
    ) -> None:
        """Same query, same mode, twice -> identical results."""
        bm25_path = tmp_path / "bm25.db"
        BM25Index.build(tiny_corpus, bm25_path)

        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        DenseIndex.build(matrix, tiny_corpus, tmp_path)

        embedder = FakeEmbedder()
        retriever = Retriever.load(tmp_path, embedder)

        # BM25 mode
        results1_bm25 = await retriever.retrieve("account", mode=RetrievalMode.BM25, top_k=5)
        results2_bm25 = await retriever.retrieve("account", mode=RetrievalMode.BM25, top_k=5)

        assert len(results1_bm25) == len(results2_bm25)
        for r1, r2 in zip(results1_bm25, results2_bm25, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank

        # Dense mode
        results1_dense = await retriever.retrieve("account", mode=RetrievalMode.DENSE, top_k=5)
        results2_dense = await retriever.retrieve("account", mode=RetrievalMode.DENSE, top_k=5)

        assert len(results1_dense) == len(results2_dense)
        for r1, r2 in zip(results1_dense, results2_dense, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank

        # Hybrid mode
        results1_hybrid = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)
        results2_hybrid = await retriever.retrieve("account", mode=RetrievalMode.HYBRID, top_k=5)

        assert len(results1_hybrid) == len(results2_hybrid)
        for r1, r2 in zip(results1_hybrid, results2_hybrid, strict=True):
            assert r1.chunk.chunk_id == r2.chunk.chunk_id
            assert r1.score == r2.score
            assert r1.rank == r2.rank
