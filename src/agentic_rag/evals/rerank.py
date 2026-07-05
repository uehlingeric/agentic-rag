"""Reranker evaluation harness against the golden dataset.

Isolates the reranker's contribution by comparing two orderings of the SAME
candidate pool: the retriever's baseline ranking (retrieval order cut to top_k)
and the reranker's reordering of that full pool (cut to top_k).

This design controls for retrieval variance: both the baseline and reranked
results use identical candidates, so any metric differences isolate the
reranker's ordering quality.

Metrics per query (same as retrieval harness):
- Recall@k: fraction of citations covered by any chunk in top-k.
- Precision@5: fraction of top-5 chunks covering >= 1 citation (always /5 denominator).
- MRR: 1 / rank of first covering chunk in top-20 (0.0 if none).
- NDCG@10: normalized discounted cumulative gain with novelty-binary gains.

Aggregation: macro-average over answerable examples only (type != "unanswerable").
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentic_rag.evals.retrieval import (
    EvalReport,
    GoldenExample,
    ModeReport,
    evaluate_ranking,
)
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import RetrievalMode

if TYPE_CHECKING:
    from agentic_rag.rerank.base import Reranker
    from agentic_rag.retrieval.retriever import Retriever


async def run_rerank_eval(
    retriever: Retriever,
    reranker: Reranker,
    golden: list[GoldenExample],
    *,
    modes: list[str],
    pool: int = 30,
    top_k: int = 10,
    config: dict[str, object] | None = None,
) -> EvalReport:
    """Run reranker evaluation over golden examples.

    Compares baseline retrieval ranking with reranked ranking on the same
    candidate pool for each mode and question.

    Args:
        retriever: Retriever instance to fetch candidate pools.
        reranker: Reranker instance to reorder candidates.
        golden: List of golden examples.
        modes: List of retrieval modes to evaluate ("bm25", "dense", "hybrid").
        pool: Number of candidates to retrieve per query (default 30).
        top_k: Top-k for ranking metrics and rerank output (default 10).
        config: Optional config dict to include in report.

    Returns:
        EvalReport with baseline and reranked ModeReports for each mode.
    """
    if config is None:
        config = {}

    # Filter to answerable examples
    answerable = [ex for ex in golden if ex.type != "unanswerable"]
    unanswerable_count = len(golden) - len(answerable)

    # Aggregate usage and timing across all rerank calls
    total_usage = Usage.zero()
    total_rerank_seconds = 0.0
    rerank_call_count = 0

    mode_reports = []
    for mode_name in modes:
        mode = RetrievalMode(mode_name)
        per_query_baseline: dict[str, dict[str, float]] = {}
        per_query_reranked: dict[str, dict[str, float]] = {}

        aggregated_baseline: dict[str, list[float]] = {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "ndcg@10": [],
        }
        aggregated_reranked: dict[str, list[float]] = {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "ndcg@10": [],
        }

        for example in answerable:
            # Retrieve candidate pool once
            candidates = await retriever.retrieve(example.question, mode=mode, top_k=pool)

            # Baseline: retrieval order cut to top_k
            baseline_ranking = candidates[:top_k]
            baseline_metrics = evaluate_ranking(example.source_citations, baseline_ranking)
            per_query_baseline[example.id] = baseline_metrics

            # Reranked: rerank full pool and cut to top_k
            start_time = time.perf_counter()
            reranked_ranking = await reranker.rerank(example.question, candidates, top_k=top_k)
            elapsed = time.perf_counter() - start_time

            reranked_metrics = evaluate_ranking(example.source_citations, reranked_ranking)
            per_query_reranked[example.id] = reranked_metrics

            # Accumulate usage and timing
            total_usage = total_usage + reranker.last_usage
            total_rerank_seconds += elapsed
            rerank_call_count += 1

            # Accumulate for averaging
            for key in aggregated_baseline:
                aggregated_baseline[key].append(baseline_metrics[key])
            for key in aggregated_reranked:
                aggregated_reranked[key].append(reranked_metrics[key])

        # Macro-average for baseline
        baseline_mode_metrics = {
            key: sum(vals) / len(vals) if vals else 0.0 for key, vals in aggregated_baseline.items()
        }

        # Macro-average for reranked
        reranked_mode_metrics = {
            key: sum(vals) / len(vals) if vals else 0.0 for key, vals in aggregated_reranked.items()
        }

        # Add both reports for this mode
        mode_reports.append(
            ModeReport(
                mode=mode_name,
                metrics=baseline_mode_metrics,
                per_query=per_query_baseline,
            )
        )
        mode_reports.append(
            ModeReport(
                mode=f"{mode_name}+{reranker.name}",
                metrics=reranked_mode_metrics,
                per_query=per_query_reranked,
            )
        )

    # Augment config with reranker info
    augmented_config: dict[str, object] = dict(config)
    augmented_config["reranker"] = reranker.name
    augmented_config["pool"] = pool
    augmented_config["top_k"] = top_k
    augmented_config["rerank_total_input_tokens"] = total_usage.input_tokens
    augmented_config["rerank_total_output_tokens"] = total_usage.output_tokens
    augmented_config["rerank_total_cost_usd"] = total_usage.cost_usd or 0.0
    mean_seconds = total_rerank_seconds / rerank_call_count if rerank_call_count > 0 else 0.0
    augmented_config["rerank_mean_seconds_per_query"] = mean_seconds

    return EvalReport(
        modes=mode_reports,
        n_answerable=len(answerable),
        n_skipped_unanswerable=unanswerable_count,
        config=augmented_config,
    )
