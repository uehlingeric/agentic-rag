"""Tests for dense retrieval index."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.providers.base import EmbeddingResult, Usage
from agentic_rag.retrieval.base import ChunkRecord
from agentic_rag.retrieval.dense import DenseIndex
from agentic_rag.retrieval.embed import EmbeddingMatrix, corpus_fingerprint


class FakeEmbedder:
    """Deterministic test embedder for reproducible tests."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        """Embed texts deterministically based on SHA256."""
        self.calls.append(texts)

        vectors = []
        for text in texts:
            # Deterministic 8-dim vector with bounded values in [-1, 1)
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = (np.frombuffer(h[:8], dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            vectors.append(vec.tolist())

        return EmbeddingResult(
            vectors=vectors,
            model=model or "fake",
            dimensions=8,
            usage=Usage(input_tokens=len(texts), output_tokens=0),
        )


class TestDenseIndex:
    """Test DenseIndex functionality."""

    def test_build_and_load(self, tmp_path: Path, tiny_corpus) -> None:
        """Build and load roundtrip preserves data."""
        # Create embedding matrix
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        # Build index
        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, tiny_corpus, index_dir)

        # Verify size and manifest
        assert index.size == 5
        assert index.manifest.model == "fake"
        assert index.manifest.dimensions == 8
        assert index.manifest.count == 5
        assert index.manifest.fingerprint == corpus_fingerprint(matrix.chunk_ids)

        # Load from disk
        index2 = DenseIndex.load(index_dir)
        assert index2.size == 5
        assert index2.manifest.model == "fake"
        assert index2.manifest.fingerprint == corpus_fingerprint(matrix.chunk_ids)

    def test_build_mismatched_chunk_ids(self, tmp_path: Path, tiny_corpus) -> None:
        """Mismatched chunk_ids between matrix and chunks raise ValueError."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        wrong_chunk_ids = ["wrong1", "wrong2", "wrong3", "wrong4", "wrong5"]
        matrix = EmbeddingMatrix(
            chunk_ids=wrong_chunk_ids,
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        with pytest.raises(ValueError, match="chunk_ids mismatch"):
            DenseIndex.build(matrix, tiny_corpus, index_dir)

    def test_load_missing_files(self, tmp_path: Path) -> None:
        """Load on missing files raises FileNotFoundError."""
        index_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Run `agentic-rag index` first"):
            DenseIndex.load(index_dir)

    @pytest.mark.asyncio
    async def test_search_with_normalization(self, tmp_path: Path, tiny_corpus) -> None:
        """Search works with L2-normalization (scaling doesn't affect ranking)."""
        # Create 4 known vectors (normalized)
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        # Use only 5 chunks
        chunks = tiny_corpus[:5]
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            model="fake-4d",
            dimensions=4,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, chunks, index_dir)

        # Create embedder that returns known query vector
        class KnownEmbedder:
            name = "fake"

            async def embed_batch(self, texts, *, model=None):
                # Return a vector close to first chunk
                query_vec = np.array([[1.0, 0.01, 0.0, 0.0]], dtype=np.float32)
                return EmbeddingResult(
                    vectors=[query_vec[0].tolist()],
                    model=model or "fake",
                    dimensions=4,
                    usage=Usage(input_tokens=1, output_tokens=0),
                )

        embedder = KnownEmbedder()
        results = await index.search("query", embedder, top_k=5)

        # First result should be closest to [1, 0, 0, 0]
        assert results[0].chunk.chunk_id == chunks[0].chunk_id
        assert results[1].chunk.chunk_id == chunks[4].chunk_id  # [0.1, 0.1, 0.1, 0.1]

    @pytest.mark.asyncio
    async def test_search_top_k_limits(self, tmp_path: Path, tiny_corpus) -> None:
        """top_k > corpus size returns only corpus size results."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()
        results = await index.search("query", embedder, top_k=100)

        # Should return 5 results, no -1 ids
        assert len(results) == 5
        assert all(r.chunk.chunk_id for r in results)

    @pytest.mark.asyncio
    async def test_search_exact_ranking(self, tmp_path: Path) -> None:
        """Known vectors produce exact cosine scores."""
        # Create 3 chunks with hand-set vectors
        chunks = [
            ChunkRecord(
                chunk_id="c1",
                doc_id="d1",
                section_id="s1",
                section_ids=["s1"],
                section_path="p",
                heading="h1",
                page_start=1,
                page_end=1,
                token_count=1,
                text="text1",
            ),
            ChunkRecord(
                chunk_id="c2",
                doc_id="d1",
                section_id="s2",
                section_ids=["s2"],
                section_path="p",
                heading="h2",
                page_start=1,
                page_end=1,
                token_count=1,
                text="text2",
            ),
            ChunkRecord(
                chunk_id="c3",
                doc_id="d1",
                section_id="s3",
                section_ids=["s3"],
                section_path="p",
                heading="h3",
                page_start=1,
                page_end=1,
                token_count=1,
                text="text3",
            ),
        ]

        # Hand-set vectors (will be normalized)
        vectors = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            model="fake",
            dimensions=2,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, chunks, index_dir)

        # Create embedder that returns [1, 0] (after normalization)
        class VecEmbedder:
            name = "fake"

            async def embed_batch(self, texts, *, model=None):
                query_vec = np.array([[1.0, 0.0]], dtype=np.float32)
                return EmbeddingResult(
                    vectors=[query_vec[0].tolist()],
                    model=model or "fake",
                    dimensions=2,
                    usage=Usage(input_tokens=1, output_tokens=0),
                )

        embedder = VecEmbedder()
        results = await index.search("query", embedder, top_k=3)

        # c1 should be highest (inner product with [1,0] is 1.0)
        # c2 should be next (~0.994 after L2 normalization of [0.9, 0.1])
        # c3 should be last (0.0)
        assert results[0].chunk.chunk_id == "c1"
        assert results[1].chunk.chunk_id == "c2"
        assert results[2].chunk.chunk_id == "c3"
        assert results[0].score == pytest.approx(1.0, abs=0.01)
        assert results[1].score == pytest.approx(0.994, abs=0.01)
        assert results[2].score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_search_prefix_nomic(self, tmp_path: Path, tiny_corpus) -> None:
        """nomic-embed model applies query prefix."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="nomic-embed-text",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()
        await index.search("test query", embedder, top_k=5)

        # Check that embedder received prefixed query
        assert len(embedder.calls) == 1
        assert embedder.calls[0][0].startswith("search_query: ")

    @pytest.mark.asyncio
    async def test_search_no_prefix_other_model(self, tmp_path: Path, tiny_corpus) -> None:
        """Other models don't apply query prefix."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="text-embedding-3-small",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()
        await index.search("test query", embedder, top_k=5)

        # Check that embedder received non-prefixed query
        assert len(embedder.calls) == 1
        assert not embedder.calls[0][0].startswith("search_")

    def test_manifest_roundtrip(self, tmp_path: Path, tiny_corpus) -> None:
        """Manifest is persisted and reloaded correctly."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="test-model",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        DenseIndex.build(matrix, tiny_corpus, index_dir)

        index2 = DenseIndex.load(index_dir)
        assert index2.manifest.model == "test-model"
        assert index2.manifest.dimensions == 8
        assert index2.manifest.count == 5

    def test_ranked_results(self, tmp_path: Path, tiny_corpus) -> None:
        """Ranked results have 1-based rank field."""
        vectors = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        matrix = EmbeddingMatrix(
            chunk_ids=[c.chunk_id for c in tiny_corpus],
            vectors=vectors,
            model="fake",
            dimensions=8,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        index_dir = tmp_path / "index"
        index = DenseIndex.build(matrix, tiny_corpus, index_dir)

        embedder = FakeEmbedder()

        # This test is not async, so we need to use a helper
        import asyncio

        results = asyncio.run(index.search("query", embedder, top_k=3))
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
