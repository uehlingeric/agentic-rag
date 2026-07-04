"""Hybrid retrieval: BM25 (SQLite FTS5), dense (FAISS), and RRF fusion."""

from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk, load_chunks
from agentic_rag.retrieval.bm25 import BM25Index
from agentic_rag.retrieval.dense import DenseIndex, DenseManifest
from agentic_rag.retrieval.embed import EmbeddingMatrix, corpus_fingerprint, embed_corpus
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_rag.retrieval.retriever import Retriever

__all__ = [
    "BM25Index",
    "ChunkRecord",
    "DenseIndex",
    "DenseManifest",
    "EmbeddingMatrix",
    "RetrievalMode",
    "Retriever",
    "ScoredChunk",
    "corpus_fingerprint",
    "embed_corpus",
    "load_chunks",
    "reciprocal_rank_fusion",
]
