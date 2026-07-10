"""Multi-query retrieval with reranking, deduplication, and proportional budgeting.

Orchestrates sequential retrieval and reranking across sub-queries, deduplicates
chunks across queries using a proportional token budget per sub-query, and builds
the final context block. Reranker.last_usage is call-scoped mutable state, so
sub-queries are processed sequentially to correctly attribute usage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_rag.agent.state import Plan, SubQueryResult
from agentic_rag.observability import set_usage_attributes, tracer
from agentic_rag.pipeline.context import BuiltContext, build_context
from agentic_rag.pipeline.pipeline import SupportsRetrieve
from agentic_rag.providers.base import Usage
from agentic_rag.rerank.base import Reranker
from agentic_rag.retrieval.base import RetrievalMode, ScoredChunk


@dataclass(frozen=True, slots=True)
class GatherResult:
    """Merged multi-query retrieval.

    ``usage`` sums reranker LLM usage across sub-queries; ``context`` markers
    are global (1..n over the merged set).
    """

    sub_results: tuple[SubQueryResult, ...]
    context: BuiltContext
    usage: Usage


async def gather(
    retriever: SupportsRetrieve,
    reranker: Reranker,
    plan: Plan,
    *,
    mode: RetrievalMode,
    candidate_pool: int,
    top_k: int,
    max_context_tokens: int,
    count_tokens: Callable[[str], int],
) -> GatherResult:
    """Retrieve, rerank, and merge results across sub-queries with budget awareness.

    Process plan.sub_queries sequentially: per sub-query, retrieve candidates
    with candidate_pool as top_k, then rerank to top_k. ``sub_results`` carries
    each sub-query's full post-rerank list; the merge then deduplicates chunks
    across sub-queries under a proportional token budget (max_context_tokens //
    len(sub_queries)) per sub-query. Each sub-query's first new chunk is always
    included even if it exceeds budget; subsequent chunks stop at the first that
    does not fit the per-query budget. Merged order is sub-query 1's selections,
    then sub-query 2's new ones, etc.

    Args:
        retriever: Retrieval interface.
        reranker: Reranking interface (processes calls sequentially to correctly
            track last_usage, which is mutable state).
        plan: Plan with sub_queries tuple.
        mode: Retrieval mode (BM25, dense, hybrid).
        candidate_pool: Top-k for retrieval (candidate pool size).
        top_k: Top-k for reranking.
        max_context_tokens: Maximum tokens for final context.
        count_tokens: Token counting function.

    Returns:
        GatherResult with sub_results, merged context, and summed usage.
    """
    sub_results: list[SubQueryResult] = []
    accumulated_usage = Usage.zero()
    budget_per_query = max_context_tokens // len(plan.sub_queries)

    # Track chunk IDs we've already selected to deduplicate across sub-queries
    selected_chunk_ids: set[str] = set()
    merged_chunks: list[ScoredChunk] = []

    # Process sub-queries sequentially so reranker.last_usage is correctly scoped
    for sub_query_index, query in enumerate(plan.sub_queries):
        # Retrieve candidates
        with tracer().start_as_current_span("rag.retrieve") as retrieve_span:
            candidates = await retriever.retrieve(query, mode=mode, top_k=candidate_pool)
            retrieve_span.set_attribute("rag.mode", mode.value)
            retrieve_span.set_attribute("rag.chunks.count", len(candidates))
            retrieve_span.set_attribute("agent.sub_query", sub_query_index)

        # Rerank to top_k
        with tracer().start_as_current_span("rag.rerank") as rerank_span:
            reranked = await reranker.rerank(query, candidates, top_k=top_k)
            rerank_span.set_attribute("rag.reranker", reranker.name)
            rerank_span.set_attribute("rag.chunks.in", len(candidates))
            rerank_span.set_attribute("rag.chunks.out", len(reranked))
            rerank_span.set_attribute("agent.sub_query", sub_query_index)
            set_usage_attributes(rerank_span, reranker.last_usage)

        # Accumulate reranker usage
        accumulated_usage = accumulated_usage + reranker.last_usage

        # SubQueryResult carries the full post-rerank list (pre-merge), per the
        # state.py contract — traces inspect per-sub-query retrieval quality here
        sub_results.append(SubQueryResult(query=query, chunks=tuple(reranked)))

        # Walk reranked chunks in order, respecting per-query budget
        running_cost = 0
        first_new_chunk_added = False

        for chunk in reranked:
            chunk_id = chunk.chunk.chunk_id

            # Dedupe: skip if already selected, but keep walking
            if chunk_id in selected_chunk_ids:
                continue

            # Format the excerpt as build_context does, with placeholder marker
            # Marker will be assigned by build_context; use a dummy position
            marker = len(merged_chunks) + 1
            excerpt = (
                f"[{marker}] {chunk.chunk.doc_id} §{chunk.chunk.section_id} — "
                f"{chunk.chunk.heading} (p.{chunk.chunk.page_start})\n"
                f"{chunk.chunk.text}\n"
            )
            excerpt_tokens = count_tokens(excerpt)

            # Always include the first new chunk even if it exceeds budget
            if not first_new_chunk_added:
                running_cost += excerpt_tokens
                merged_chunks.append(chunk)
                selected_chunk_ids.add(chunk_id)
                first_new_chunk_added = True
            elif running_cost + excerpt_tokens <= budget_per_query:
                running_cost += excerpt_tokens
                merged_chunks.append(chunk)
                selected_chunk_ids.add(chunk_id)
            else:
                # This chunk doesn't fit; stop walking for this sub-query
                break

    # Build final context from merged chunks
    context = build_context(merged_chunks, max_tokens=max_context_tokens, count_tokens=count_tokens)

    return GatherResult(sub_results=tuple(sub_results), context=context, usage=accumulated_usage)
