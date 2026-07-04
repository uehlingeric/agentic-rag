"""Resumable batch embedding pipeline with checkpointing.

Embeds chunks via an embedding provider, with checkpoint support for resuming
interrupted runs. Checkpoint files are JSONL with a header line and one vector
line per embedded chunk, flushed after each batch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ChunkRecord

if TYPE_CHECKING:
    from agentic_rag.providers.base import EmbeddingProvider


def corpus_fingerprint(chunk_ids: Sequence[str]) -> str:
    """Order-sensitive fingerprint identifying a corpus for checkpoint/index reuse."""
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()


def doc_prefix(model: str) -> str:
    """Get document prefix for a model string.

    nomic-embed models require task prefixes for retrieval quality.
    Other models do not use them.
    """
    if "nomic-embed" in model:
        return "search_document: "
    return ""


def query_prefix(model: str) -> str:
    """Get query prefix for a model string.

    nomic-embed models require task prefixes for retrieval quality.
    Other models do not use them.
    """
    if "nomic-embed" in model:
        return "search_query: "
    return ""


@dataclass(frozen=True, slots=True)
class EmbeddingMatrix:
    """Dense embedding vectors with metadata.

    ``vectors`` has shape (n, d) and is NOT pre-normalized (normalization
    happens downstream in dense.py for cosine similarity via IndexFlatIP).
    """

    chunk_ids: list[str]
    vectors: npt.NDArray[np.float32]
    model: str
    dimensions: int
    usage: Usage


async def embed_corpus(
    chunks: Sequence[ChunkRecord],
    embedder: EmbeddingProvider,
    *,
    model: str,
    checkpoint_path: Path,
    batch_size: int = 32,
    force: bool = False,
) -> EmbeddingMatrix:
    """Embed chunks with checkpointing support for resumable runs.

    Args:
        chunks: Sequence of ChunkRecord objects to embed.
        embedder: EmbeddingProvider instance.
        model: Model string passed to embedder.embed_batch.
        checkpoint_path: Path to JSONL checkpoint file.
        batch_size: Number of chunks per batch.
        force: If True, delete any existing checkpoint and start fresh.

    Returns:
        EmbeddingMatrix with vectors in chunks order, combining checkpoint
        and fresh embeddings.

    Raises:
        ValueError: If embeddings have inconsistent dimensions.
    """
    chunk_ids = [c.chunk_id for c in chunks]
    fingerprint = corpus_fingerprint(chunk_ids)

    # Load or initialize checkpoint state
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if force:
        checkpoint_path.unlink(missing_ok=True)

    cached_vectors: dict[str, list[float]] = {}
    header_valid = False

    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    row = json.loads(line)
                    if (
                        row.get("kind") == "header"
                        and row.get("model") == model
                        and row.get("fingerprint") == fingerprint
                    ):
                        header_valid = True
                else:
                    row = json.loads(line)
                    if row.get("kind") == "vec":
                        cached_vectors[row["chunk_id"]] = row["vector"]

        if not header_valid:
            sys.stderr.write(f"Checkpoint header mismatch; starting fresh at {checkpoint_path}\n")
            checkpoint_path.unlink()
            cached_vectors.clear()

    # Determine which chunks need embedding
    missing_indices = [i for i, chunk in enumerate(chunks) if chunk.chunk_id not in cached_vectors]

    total_usage = Usage.zero()
    embedded_count = 0

    if missing_indices:
        # If checkpoint was valid, append mode; otherwise truncate
        if not header_valid:
            checkpoint_path.unlink(missing_ok=True)
            with checkpoint_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "kind": "header",
                            "model": model,
                            "fingerprint": fingerprint,
                        }
                    )
                    + "\n"
                )
        elif not cached_vectors:
            # Write header only if file didn't exist before
            with checkpoint_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "kind": "header",
                            "model": model,
                            "fingerprint": fingerprint,
                        }
                    )
                    + "\n"
                )

        # Process missing chunks in batches
        for batch_start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[batch_start : batch_start + batch_size]
            batch_chunks = [chunks[i] for i in batch_indices]

            # Prepare texts with prefix
            prefix = doc_prefix(model)
            texts = [prefix + chunk.heading + "\n" + chunk.text for chunk in batch_chunks]

            # Embed batch
            result = await embedder.embed_batch(texts, model=model)
            total_usage = total_usage + result.usage

            # Save to checkpoint
            with checkpoint_path.open("a", encoding="utf-8") as f:
                for chunk, vector in zip(batch_chunks, result.vectors, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "kind": "vec",
                                "chunk_id": chunk.chunk_id,
                                "vector": vector,
                            }
                        )
                        + "\n"
                    )
                    cached_vectors[chunk.chunk_id] = vector

            embedded_count += len(batch_chunks)
            progress = len(cached_vectors)
            total = len(chunks)
            sys.stderr.write(f"embedding: {progress}/{total}\n")
            sys.stderr.flush()

    # Assemble final matrix in chunks order
    vectors_list = [cached_vectors[cid] for cid in chunk_ids]

    if vectors_list:
        dims = {len(v) for v in vectors_list}
        if len(dims) > 1:
            raise ValueError("Embeddings have inconsistent dimensions")
        dimensions = dims.pop()
    else:
        dimensions = 0

    vectors_array = np.array(vectors_list, dtype=np.float32)

    return EmbeddingMatrix(
        chunk_ids=chunk_ids,
        vectors=vectors_array,
        model=model,
        dimensions=dimensions,
        usage=total_usage,
    )
