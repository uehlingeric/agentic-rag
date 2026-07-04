"""Orchestrate corpus ingestion: download, extract, chunk, serialize."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.ingest.chunk import chunk_sections
from agentic_rag.ingest.download import download_corpus
from agentic_rag.ingest.extract import extract_sections
from agentic_rag.ingest.sources import SOURCES


@dataclass
class DocumentReport:
    """Statistics for a processed document."""

    doc_id: str
    sha256: str
    pages: int
    sections: int
    chunks: int


@dataclass
class IngestManifest:
    """Manifest of ingested corpus with output metadata."""

    documents: list[DocumentReport]
    total_chunks: int
    output_path: Path
    target_tokens: int
    overlap_tokens: int
    tokenizer: str
    created_at: str
    corpus_version: str


def run_ingest(
    settings: Settings, doc_ids: list[str] | None = None, force: bool = False
) -> IngestManifest:
    """Run full ingestion pipeline: download -> extract -> chunk -> serialize.

    When doc_ids subset is given, only those docs are (re)processed, but chunks.jsonl
    is rewritten for the subset only. Use full doc_ids list for production runs.

    Args:
        settings: Application settings.
        doc_ids: Restrict to specific doc_ids (default: all in SOURCES).
        force: Re-download and re-chunk.

    Returns:
        IngestManifest with statistics and output path.
    """
    corpus_dir = settings.data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = corpus_dir / "chunks.jsonl"
    manifest_path = corpus_dir / "manifest.json"

    # Download documents
    downloaded = download_corpus(settings, doc_ids=doc_ids, force=force)

    # Track all chunks and document stats
    all_chunks = []
    doc_reports = []

    # Process in SOURCES order for determinism (use SOURCES for ordering, fall back for
    # unknown docs)
    all_doc_ids = list(SOURCES.keys()) + [d.doc_id for d in downloaded if d.doc_id not in SOURCES]
    for source_id in all_doc_ids:
        if doc_ids and source_id not in doc_ids:
            continue

        # Find downloaded doc
        downloaded_doc = next((d for d in downloaded if d.doc_id == source_id), None)
        if not downloaded_doc:
            continue

        # Extract sections
        sections = extract_sections(downloaded_doc.path, source_id)

        # Chunk sections
        chunks = chunk_sections(
            source_id,
            sections,
            target_tokens=settings.chunking.target_tokens,
            overlap_tokens=settings.chunking.overlap_tokens,
        )

        all_chunks.extend(chunks)

        # Record stats
        num_pages = _count_pages(sections)
        doc_reports.append(
            DocumentReport(
                doc_id=source_id,
                sha256=downloaded_doc.sha256,
                pages=num_pages,
                sections=len(sections),
                chunks=len(chunks),
            )
        )

    # Write chunks.jsonl with ordered fields
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            # Write fields in explicit order for consistency
            chunk_obj = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "section_ids": chunk.section_ids,
                "section_path": chunk.section_path,
                "heading": chunk.heading,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "content_type": chunk.content_type,
                "text": chunk.text,
            }
            json.dump(chunk_obj, f, ensure_ascii=False)
            f.write("\n")

    # Create and write manifest
    created_at_iso = datetime.now(UTC).isoformat()
    manifest = IngestManifest(
        documents=doc_reports,
        total_chunks=len(all_chunks),
        output_path=chunks_path,
        target_tokens=settings.chunking.target_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
        tokenizer="o200k_base",
        created_at=created_at_iso,
        corpus_version="v1",
    )

    manifest_dict = asdict(manifest)
    manifest_dict["output_path"] = str(manifest_dict["output_path"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)

    return manifest


def _count_pages(sections: list) -> int:
    """Count unique pages across sections."""
    if not sections:
        return 0
    return sections[-1].page_end + 1
