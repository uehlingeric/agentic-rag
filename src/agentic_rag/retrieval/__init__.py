"""Hybrid retrieval: BM25 (SQLite FTS5), dense (FAISS), and RRF fusion."""

from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk, load_chunks

__all__ = ["ChunkRecord", "RetrievalMode", "ScoredChunk", "load_chunks"]
