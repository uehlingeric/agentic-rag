"""Reciprocal rank fusion for combining multiple ranked retrieval results.

Fuses results from multiple retrieval modes (BM25, dense, etc.) into a single
ranked list using reciprocal rank fusion (RRF) scoring.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from agentic_rag.retrieval.base import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]], *, k: int = 60
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists using reciprocal rank fusion.

    Args:
        rankings: Mapping of mode names to ranked lists of ScoredChunk.
                  Keys are mode names (e.g., "bm25", "dense"); values are
                  ranked result sequences.
        k: RRF parameter (default 60). Score for a chunk at rank r is
           1 / (k + r), where ranks are 1-based.

    Returns:
        List of ScoredChunk results sorted by fused score DESC, ties broken by
        chunk_id ASC. Ranks reassigned 1..n. source_scores merged from all
        appearances of each chunk across modes.
    """
    if not rankings:
        return []

    # Accumulate scores and source_scores by chunk_id
    chunk_scores: dict[str, float] = defaultdict(float)
    chunk_sources: dict[str, dict[str, float]] = defaultdict(dict)
    chunk_objects: dict[str, ScoredChunk] = {}

    for _mode_name, ranked_list in rankings.items():
        for scored_chunk in ranked_list:
            chunk_id = scored_chunk.chunk.chunk_id
            # RRF score: 1 / (k + rank)
            score_contribution = 1.0 / (k + scored_chunk.rank)
            chunk_scores[chunk_id] += score_contribution
            # Merge source_scores from this chunk's appearance
            chunk_sources[chunk_id].update(scored_chunk.source_scores)
            # Keep first occurrence of the chunk object (same for all modes)
            if chunk_id not in chunk_objects:
                chunk_objects[chunk_id] = scored_chunk

    if not chunk_scores:
        return []

    # Sort by fused score DESC, then by chunk_id ASC for determinism
    sorted_ids = sorted(
        chunk_scores.keys(),
        key=lambda cid: (-chunk_scores[cid], cid),
    )

    # Build result list with reassigned ranks
    result = []
    for new_rank, chunk_id in enumerate(sorted_ids, start=1):
        original_chunk_obj = chunk_objects[chunk_id]
        fused_chunk = ScoredChunk(
            chunk=original_chunk_obj.chunk,
            score=chunk_scores[chunk_id],
            rank=new_rank,
            source_scores=chunk_sources[chunk_id],
        )
        result.append(fused_chunk)

    return result
