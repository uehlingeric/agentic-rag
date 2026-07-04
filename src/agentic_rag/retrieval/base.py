"""Shared retrieval contracts: chunk records, scored results, and modes.

This module is the frozen contract between the index implementations
(``bm25.py``, ``dense.py``), fusion (``fusion.py``), the unified
``Retriever``, and the eval harness. Downstream code depends only on these
types — never on index internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One retrievable chunk, mirroring the fields of ``chunks.jsonl``.

    ``section_id`` is the primary section; ``section_ids`` carries every
    section merged into the chunk and is the key citation matching operates on.
    """

    chunk_id: str
    doc_id: str
    section_id: str
    section_ids: list[str]
    section_path: str
    heading: str
    page_start: int
    page_end: int
    token_count: int
    text: str

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> ChunkRecord:
        """Build a record from one parsed JSONL row, ignoring unknown keys."""
        return cls(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            section_ids=list(row["section_ids"]),
            section_path=row["section_path"],
            heading=row["heading"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            token_count=row["token_count"],
            text=row["text"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "section_ids": self.section_ids,
            "section_path": self.section_path,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with its retrieval score.

    ``rank`` is 1-based within the result list. ``source_scores`` maps mode
    name -> that mode's raw score; hybrid results carry one entry per mode
    that surfaced the chunk, single-mode results carry their own raw score.
    """

    chunk: ChunkRecord
    score: float
    rank: int
    source_scores: dict[str, float] = field(default_factory=dict)


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Load all chunks from a JSONL corpus, preserving file order."""
    if not path.exists():
        msg = f"Chunk corpus not found at {path}. Run `agentic-rag ingest` first."
        raise FileNotFoundError(msg)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ChunkRecord.from_json(json.loads(line)))
    return records
