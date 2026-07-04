"""Tests for embedding pipeline with checkpointing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.providers.base import EmbeddingResult, Usage
from agentic_rag.retrieval.embed import (
    doc_prefix,
    embed_corpus,
    query_prefix,
)


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


class TestPrefixes:
    """Test prefix helper functions."""

    def test_doc_prefix_nomic(self) -> None:
        """nomic-embed models get document prefix."""
        assert doc_prefix("nomic-embed-text") == "search_document: "
        assert doc_prefix("nomic-embed-text-v1.5") == "search_document: "

    def test_doc_prefix_other(self) -> None:
        """Other models don't get prefix."""
        assert doc_prefix("text-embedding-3-small") == ""
        assert doc_prefix("openai") == ""

    def test_query_prefix_nomic(self) -> None:
        """nomic-embed models get query prefix."""
        assert query_prefix("nomic-embed-text") == "search_query: "

    def test_query_prefix_other(self) -> None:
        """Other models don't get prefix."""
        assert query_prefix("text-embedding-3-small") == ""


class TestEmbedCorpus:
    """Test embed_corpus function."""

    @pytest.mark.asyncio
    async def test_full_run(self, tmp_path: Path, tiny_corpus) -> None:
        """Full embedding run produces correct matrix."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        matrix = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=2,
        )

        # Validate matrix
        assert matrix.chunk_ids == [c.chunk_id for c in tiny_corpus]
        assert matrix.vectors.shape == (5, 8)
        assert matrix.model == "fake"
        assert matrix.dimensions == 8
        assert matrix.usage.input_tokens == 5

        # Validate checkpoint structure
        with checkpoint.open(encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 6  # header + 5 vectors
        header = json.loads(lines[0])
        assert header["kind"] == "header"
        assert header["model"] == "fake"

    @pytest.mark.asyncio
    async def test_resume(self, tmp_path: Path, tiny_corpus) -> None:
        """Resume from checkpoint embeds only missing chunks."""
        embedder1 = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        # Embed all chunks
        full_matrix = await embed_corpus(
            tiny_corpus,
            embedder1,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Pre-write checkpoint with first 2 chunks
        checkpoint.unlink()
        chunk_ids = [c.chunk_id for c in tiny_corpus]
        fingerprint = hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()

        with checkpoint.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "header",
                        "model": "fake",
                        "fingerprint": fingerprint,
                    }
                )
                + "\n"
            )
            # Add first 2 chunks
            for chunk in tiny_corpus[:2]:
                prefix = ""
                text = prefix + chunk.heading + "\n" + chunk.text
                h = hashlib.sha256(text.encode("utf-8")).digest()
                vec = (
                    (np.frombuffer(h[:8], dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                ).tolist()
                f.write(
                    json.dumps(
                        {
                            "kind": "vec",
                            "chunk_id": chunk.chunk_id,
                            "vector": vec,
                        }
                    )
                    + "\n"
                )

        # Resume from checkpoint
        embedder2 = FakeEmbedder()
        resumed_matrix = await embed_corpus(
            tiny_corpus,
            embedder2,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Should only embed the missing 3 chunks
        assert len(embedder2.calls) == 1
        assert len(embedder2.calls[0]) == 3
        assert embedder2.calls[0] == [
            tiny_corpus[2].heading + "\n" + tiny_corpus[2].text,
            tiny_corpus[3].heading + "\n" + tiny_corpus[3].text,
            tiny_corpus[4].heading + "\n" + tiny_corpus[4].text,
        ]

        # Final matrix matches full run
        np.testing.assert_array_almost_equal(resumed_matrix.vectors, full_matrix.vectors)
        assert resumed_matrix.usage.input_tokens == 3

    @pytest.mark.asyncio
    async def test_fingerprint_mismatch(self, tmp_path: Path, tiny_corpus) -> None:
        """Fingerprint mismatch forces re-embed."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        # Write checkpoint with wrong fingerprint
        with checkpoint.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "header",
                        "model": "fake",
                        "fingerprint": "wrongfp",
                    }
                )
                + "\n"
            )

        matrix = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # All chunks re-embedded
        assert matrix.usage.input_tokens == 5

    @pytest.mark.asyncio
    async def test_model_mismatch(self, tmp_path: Path, tiny_corpus) -> None:
        """Model mismatch in header forces re-embed."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        chunk_ids = [c.chunk_id for c in tiny_corpus]
        fingerprint = hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()

        # Write checkpoint with different model
        with checkpoint.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "header",
                        "model": "other-model",
                        "fingerprint": fingerprint,
                    }
                )
                + "\n"
            )

        matrix = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # All chunks re-embedded
        assert matrix.usage.input_tokens == 5

    @pytest.mark.asyncio
    async def test_force_flag(self, tmp_path: Path, tiny_corpus) -> None:
        """force=True deletes checkpoint and re-embeds."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        # Embed once
        matrix1 = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Embed again with force=True
        matrix2 = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
            force=True,
        )

        # Second run embedded all 5
        assert matrix2.usage.input_tokens == 5
        # Matrices should be identical
        np.testing.assert_array_almost_equal(matrix1.vectors, matrix2.vectors)

    @pytest.mark.asyncio
    async def test_nomic_prefix(self, tmp_path: Path, tiny_corpus) -> None:
        """nomic-embed model adds document prefix."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        await embed_corpus(
            tiny_corpus,
            embedder,
            model="nomic-embed-text",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Check that texts have prefix
        for call in embedder.calls:
            for text in call:
                assert text.startswith("search_document: ")

    @pytest.mark.asyncio
    async def test_non_nomic_no_prefix(self, tmp_path: Path, tiny_corpus) -> None:
        """Non-nomic model doesn't add prefix."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        await embed_corpus(
            tiny_corpus,
            embedder,
            model="text-embedding-3-small",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Check that texts don't have prefix
        for call in embedder.calls:
            for text in call:
                assert not text.startswith("search_")

    @pytest.mark.asyncio
    async def test_usage_sum(self, tmp_path: Path, tiny_corpus) -> None:
        """Usage is summed over fresh calls only (not resumed)."""
        embedder = FakeEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        chunk_ids = [c.chunk_id for c in tiny_corpus]
        fingerprint = hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()

        # Pre-write checkpoint with first 2 chunks
        with checkpoint.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "header",
                        "model": "fake",
                        "fingerprint": fingerprint,
                    }
                )
                + "\n"
            )
            for chunk in tiny_corpus[:2]:
                prefix = ""
                text = prefix + chunk.heading + "\n" + chunk.text
                h = hashlib.sha256(text.encode("utf-8")).digest()
                vec = (
                    (np.frombuffer(h[:8], dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                ).tolist()
                f.write(
                    json.dumps(
                        {
                            "kind": "vec",
                            "chunk_id": chunk.chunk_id,
                            "vector": vec,
                        }
                    )
                    + "\n"
                )

        embedder = FakeEmbedder()
        matrix = await embed_corpus(
            tiny_corpus,
            embedder,
            model="fake",
            checkpoint_path=checkpoint,
            batch_size=10,
        )

        # Only 3 new chunks embedded
        assert matrix.usage.input_tokens == 3

    @pytest.mark.asyncio
    async def test_inconsistent_dimensions(self, tmp_path: Path) -> None:
        """Inconsistent vector dimensions raise ValueError."""
        from agentic_rag.retrieval.base import ChunkRecord

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
        ]

        class BadEmbedder:
            name = "bad"

            async def embed_batch(self, texts, *, model=None):
                # Return vectors of different dimensions
                return EmbeddingResult(
                    vectors=[[1.0, 2.0], [3.0, 4.0, 5.0]],
                    model=model or "bad",
                    dimensions=3,
                    usage=Usage(input_tokens=2, output_tokens=0),
                )

        embedder = BadEmbedder()
        checkpoint = tmp_path / "vectors.jsonl"

        with pytest.raises(ValueError, match="inconsistent dimensions"):
            await embed_corpus(
                chunks,
                embedder,
                model="bad",
                checkpoint_path=checkpoint,
            )
