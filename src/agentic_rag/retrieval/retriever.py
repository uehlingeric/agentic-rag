"""Unified retrieval interface supporting BM25, dense, and hybrid modes.

Provides a single Retriever class that dispatches to BM25 and dense indexes
and fuses results using reciprocal rank fusion for hybrid retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentic_rag.retrieval.base import RetrievalMode, ScoredChunk
from agentic_rag.retrieval.bm25 import BM25Index
from agentic_rag.retrieval.dense import DenseIndex
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from agentic_rag.providers.base import EmbeddingProvider


class Retriever:
    """Unified retrieval interface for multiple modes: BM25, dense, hybrid.

    Dispatches queries to appropriate index and optionally fuses results
    from multiple modes.
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        embedder: EmbeddingProvider,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> None:
        """Initialize Retriever with indexes and embedding provider.

        Args:
            bm25: BM25Index instance.
            dense: DenseIndex instance.
            embedder: EmbeddingProvider for dense search and query embedding.
            rrf_k: RRF parameter for fusion (default 60).
            candidate_pool: Number of candidates to fetch from each mode
                            before fusion in hybrid mode (default 50).
        """
        self._bm25 = bm25
        self._dense = dense
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedder: EmbeddingProvider,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> Retriever:
        """Load Retriever from persisted indexes.

        Loads BM25 index from index_dir/bm25.db and dense index from
        index_dir (containing faiss.bin, id_map.parquet, manifest.json).

        Args:
            index_dir: Directory containing both indexes.
            embedder: EmbeddingProvider instance.
            rrf_k: RRF parameter (default 60).
            candidate_pool: Candidate pool size (default 50).

        Returns:
            Retriever instance.

        Raises:
            FileNotFoundError: If any index files are missing.
        """
        bm25_path = index_dir / "bm25.db"
        bm25 = BM25Index(bm25_path)
        dense = DenseIndex.load(index_dir)

        return cls(bm25, dense, embedder, rrf_k=rrf_k, candidate_pool=candidate_pool)

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Retrieve chunks matching the query.

        Args:
            query: Query string.
            mode: RetrievalMode (BM25, DENSE, or HYBRID). Default HYBRID.
            top_k: Maximum results to return.

        Returns:
            List of ScoredChunk results ranked by mode-specific or fused scores,
            with ranks reassigned 1..top_k.
        """
        if mode == RetrievalMode.BM25:
            return self._bm25.search(query, top_k=top_k)

        if mode == RetrievalMode.DENSE:
            return await self._dense.search(query, self._embedder, top_k=top_k)

        # HYBRID mode: fetch from both, fuse, cut to top_k, reassign ranks
        bm25_candidates = self._bm25.search(query, top_k=self._candidate_pool)
        dense_candidates = await self._dense.search(
            query, self._embedder, top_k=self._candidate_pool
        )

        fused = reciprocal_rank_fusion(
            {"bm25": bm25_candidates, "dense": dense_candidates},
            k=self._rrf_k,
        )

        # Cut to top_k and reassign ranks
        fused_cut = fused[:top_k]
        result = [
            ScoredChunk(
                chunk=sc.chunk,
                score=sc.score,
                rank=idx,
                source_scores=sc.source_scores,
            )
            for idx, sc in enumerate(fused_cut, start=1)
        ]

        return result
