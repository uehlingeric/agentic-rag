"""FAISS-based dense retrieval index with cosine similarity search.

Builds and searches a FAISS IndexFlatIP over L2-normalized embeddings,
combining with chunk metadata for result ranking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk
from agentic_rag.retrieval.embed import EmbeddingMatrix, corpus_fingerprint, query_prefix

if TYPE_CHECKING:
    from agentic_rag.providers.base import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class DenseManifest:
    """Metadata for a dense index.

    ``fingerprint`` matches the corpus fingerprint from embed_corpus.
    """

    model: str
    dimensions: int
    count: int
    fingerprint: str


class DenseIndex:
    """FAISS-based dense retrieval index with L2-normalized cosine search."""

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        id_map: pa.Table,
        manifest: DenseManifest,
    ) -> None:
        """Initialize a DenseIndex.

        Args:
            index: FAISS IndexFlatIP instance.
            id_map: PyArrow Table with chunk metadata.
            manifest: DenseManifest with index metadata.
        """
        self._index = index
        self._id_map = id_map
        self._manifest = manifest

    @classmethod
    def build(
        cls, matrix: EmbeddingMatrix, chunks: Sequence[ChunkRecord], index_dir: Path
    ) -> DenseIndex:
        """Build a new dense index from an embedding matrix.

        Args:
            matrix: EmbeddingMatrix from embed_corpus.
            chunks: Sequence of ChunkRecord objects (order must match matrix).
            index_dir: Directory where index files will be written.

        Returns:
            A new DenseIndex instance.

        Raises:
            ValueError: If matrix.chunk_ids does not match chunks in order.
        """
        # Validate order
        chunk_ids = [c.chunk_id for c in chunks]
        if matrix.chunk_ids != chunk_ids:
            raise ValueError(
                f"matrix.chunk_ids mismatch: expected {chunk_ids}, got {matrix.chunk_ids}"
            )

        index_dir.mkdir(parents=True, exist_ok=True)

        # L2-normalize vectors (copy, preserve zero-norm rows)
        vectors = matrix.vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.where(norms > 0, vectors / norms, vectors)

        # Build FAISS index
        faiss_index = faiss.IndexFlatIP(matrix.dimensions)
        faiss_index.add(normalized)

        # Write FAISS index
        faiss_path = index_dir / "faiss.bin"
        faiss.write_index(faiss_index, str(faiss_path))

        # Write id_map parquet
        id_map_dict = {
            "chunk_id": [c.chunk_id for c in chunks],
            "doc_id": [c.doc_id for c in chunks],
            "section_id": [c.section_id for c in chunks],
            "section_ids": [c.section_ids for c in chunks],
            "section_path": [c.section_path for c in chunks],
            "heading": [c.heading for c in chunks],
            "page_start": [c.page_start for c in chunks],
            "page_end": [c.page_end for c in chunks],
            "token_count": [c.token_count for c in chunks],
            "text": [c.text for c in chunks],
        }
        id_map_table = pa.table(id_map_dict)
        id_map_path = index_dir / "id_map.parquet"
        pq.write_table(id_map_table, str(id_map_path))

        # Write manifest
        manifest = DenseManifest(
            model=matrix.model,
            dimensions=matrix.dimensions,
            count=len(chunks),
            fingerprint=corpus_fingerprint(matrix.chunk_ids),
        )

        manifest_dict = asdict(manifest)
        manifest_path = index_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, sort_keys=True)
            f.write("\n")

        # Reload to validate
        return cls.load(index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> DenseIndex:
        """Load a dense index from disk.

        Args:
            index_dir: Directory containing faiss.bin, id_map.parquet,
                      and manifest.json.

        Returns:
            A DenseIndex instance.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        faiss_path = index_dir / "faiss.bin"
        id_map_path = index_dir / "id_map.parquet"
        manifest_path = index_dir / "manifest.json"

        for path in [faiss_path, id_map_path, manifest_path]:
            if not path.exists():
                msg = f"Dense index file missing: {path}. Run `agentic-rag index` first."
                raise FileNotFoundError(msg)

        # Load FAISS index
        faiss_index = cast(faiss.IndexFlatIP, faiss.read_index(str(faiss_path)))

        # Load id_map
        id_map = pq.read_table(str(id_map_path))

        # Load manifest
        with manifest_path.open(encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest = DenseManifest(**manifest_dict)

        # Validate
        if manifest.count != id_map.num_rows:
            raise ValueError(
                f"Manifest count {manifest.count} does not match id_map rows {id_map.num_rows}"
            )
        if manifest.dimensions != faiss_index.d:
            raise ValueError(
                f"Manifest dimensions {manifest.dimensions} does not match "
                f"FAISS dimensions {faiss_index.d}"
            )

        return cls(faiss_index, id_map, manifest)

    async def search(
        self, query: str, embedder: EmbeddingProvider, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Search the index with a text query.

        Args:
            query: Query text.
            embedder: EmbeddingProvider instance.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk results, ranked by cosine similarity (1-based).
        """
        # Embed query with prefix
        prefix = query_prefix(self._manifest.model)
        query_text = prefix + query
        result = await embedder.embed_batch([query_text], model=self._manifest.model)

        # Get query vector and normalize
        query_vector = np.array(result.vectors[0], dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / norm if norm[0, 0] > 0 else query_vector

        # Search FAISS
        distances, indices = self._index.search(query_vector, top_k)

        # Convert to ScoredChunk
        scored_chunks: list[ScoredChunk] = []
        for rank, (idx, distance) in enumerate(
            zip(indices[0], distances[0], strict=False), start=1
        ):
            if idx == -1:  # FAISS pads with -1 when top_k > ntotal; padding is terminal
                break

            row = self._id_map.slice(int(idx), 1)
            chunk = ChunkRecord(
                chunk_id=row["chunk_id"][0].as_py(),
                doc_id=row["doc_id"][0].as_py(),
                section_id=row["section_id"][0].as_py(),
                section_ids=row["section_ids"][0].as_py(),
                section_path=row["section_path"][0].as_py(),
                heading=row["heading"][0].as_py(),
                page_start=int(row["page_start"][0].as_py()),
                page_end=int(row["page_end"][0].as_py()),
                token_count=int(row["token_count"][0].as_py()),
                text=row["text"][0].as_py(),
            )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(distance),
                    rank=rank,
                    source_scores={"dense": float(distance)},
                )
            )

        return scored_chunks

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return self._manifest.count

    @property
    def manifest(self) -> DenseManifest:
        """Index metadata."""
        return self._manifest
