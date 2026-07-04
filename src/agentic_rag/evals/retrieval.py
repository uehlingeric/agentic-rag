"""Retrieval evaluation harness against the golden dataset.

Metrics per query:
- Recall@k: fraction of citations covered by any chunk in top-k.
- Precision@5: fraction of top-5 chunks covering >= 1 citation (always /5 denominator).
- MRR: 1 / rank of first covering chunk in top-20 (0.0 if none).
- NDCG@10: normalized discounted cumulative gain with novelty-binary gains:
  rel_i = 1 only if the chunk at rank i covers at least one citation not already
  covered at an earlier rank (repeat coverage of the same citation earns nothing).
  DCG = sum(rel_i / log2(i+1)) for i=1..10; IDCG = sum(1/log2(i+1)) for
  i=1..min(len(citations), 10) — the ideal ranking surfaces each citation once,
  one chunk per citation, at the top. This keeps NDCG in [0, 1] even when a
  section spans many chunks that all match the same citation.

Aggregation: macro-average over answerable examples only (type != "unanswerable").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk

if TYPE_CHECKING:
    from agentic_rag.retrieval.retriever import Retriever


@dataclass(frozen=True, slots=True)
class Citation:
    """One source citation from the golden dataset."""

    doc: str
    section: str


@dataclass(frozen=True, slots=True)
class GoldenExample:
    """One golden question with its reference answer and source citations."""

    id: str
    question: str
    reference_answer: str
    source_citations: list[Citation]
    difficulty: str
    type: str


def load_golden(path: Path) -> list[GoldenExample]:
    """Load golden examples from a JSONL file.

    Args:
        path: Path to golden.jsonl.

    Returns:
        List of GoldenExample, parsed in file order.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If any row has invalid schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    examples = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Validate schema
                required = {
                    "id",
                    "question",
                    "reference_answer",
                    "source_citations",
                    "difficulty",
                    "type",
                }
                if not required.issubset(row.keys()):
                    msg = f"Line {line_no}: missing keys. Required: {required}"
                    raise ValueError(msg)

                citations = [
                    Citation(doc=c["doc"], section=c["section"]) for c in row["source_citations"]
                ]
                examples.append(
                    GoldenExample(
                        id=row["id"],
                        question=row["question"],
                        reference_answer=row["reference_answer"],
                        source_citations=citations,
                        difficulty=row["difficulty"],
                        type=row["type"],
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                msg = f"Line {line_no}: {e}"
                raise ValueError(msg) from e

    return examples


def covers(chunk: ChunkRecord, citation: Citation) -> bool:
    """Check if a chunk covers a citation.

    Args:
        chunk: Chunk record.
        citation: Citation to check.

    Returns:
        True if chunk.doc_id == citation.doc and citation.section in chunk.section_ids.
    """
    return chunk.doc_id == citation.doc and citation.section in chunk.section_ids


def evaluate_ranking(citations: list[Citation], ranking: list[ScoredChunk]) -> dict[str, float]:
    """Compute retrieval metrics for a single query.

    Args:
        citations: Expected citations for the query.
        ranking: Ranked list of retrieved chunks (any length >= 0).

    Returns:
        Dict with keys: recall@5, recall@10, recall@20, precision@5, mrr, ndcg@10.
        All values are floats. Metrics for which no covering chunks exist return 0.0.
    """
    if not citations:
        # No citations means no recall/ndcg; mrr=0, precision@5=0
        return {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "precision@5": 0.0,
            "mrr": 0.0,
            "ndcg@10": 0.0,
        }

    # Recall: set of citations covered by any chunk at each cutoff
    covered_at_k: dict[int, set[int]] = {5: set(), 10: set(), 20: set()}
    for i, scored in enumerate(ranking):
        if i >= 20:
            break
        for cit_idx, citation in enumerate(citations):
            if covers(scored.chunk, citation):
                if i < 5:
                    covered_at_k[5].add(cit_idx)
                if i < 10:
                    covered_at_k[10].add(cit_idx)
                covered_at_k[20].add(cit_idx)

    n_citations = len(citations)
    recall_5 = len(covered_at_k[5]) / n_citations
    recall_10 = len(covered_at_k[10]) / n_citations
    recall_20 = len(covered_at_k[20]) / n_citations

    # Precision@5: fraction of top-5 chunks that cover >= 1 citation
    top_5 = ranking[:5]
    n_relevant_in_top_5 = sum(
        1 for scored in top_5 if any(covers(scored.chunk, cit) for cit in citations)
    )
    precision_5 = n_relevant_in_top_5 / 5  # Always /5 even if fewer results

    # MRR: 1 / rank of first covering chunk in top-20, 0 if none
    mrr = 0.0
    for i, scored in enumerate(ranking[:20]):
        if any(covers(scored.chunk, cit) for cit in citations):
            mrr = 1.0 / (i + 1)
            break

    # NDCG@10: DCG / IDCG with novelty-binary gains — a chunk only earns gain
    # when it covers a citation not already covered at an earlier rank, so
    # repeated coverage of one citation cannot push NDCG above 1.0.
    dcg = 0.0
    seen: set[int] = set()
    for i, scored in enumerate(ranking[:10]):
        new = {
            ci for ci, cit in enumerate(citations) if ci not in seen and covers(scored.chunk, cit)
        }
        if new:
            dcg += 1.0 / math.log2(i + 2)  # rank i+1, so log2((i+1)+1) = log2(i+2)
            seen |= new

    # IDCG: sum of 1 / log2(r+1) for rank r=1..min(len(citations), 10)
    # In 0-based terms: sum(1 / log2(i+2)) for i=0..min(len(citations)-1, 9)
    n_ideal = min(n_citations, 10)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_ideal))

    ndcg_10 = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall@5": recall_5,
        "recall@10": recall_10,
        "recall@20": recall_20,
        "precision@5": precision_5,
        "mrr": mrr,
        "ndcg@10": ndcg_10,
    }


@dataclass(frozen=True, slots=True)
class ModeReport:
    """Evaluation results for a single retrieval mode."""

    mode: str
    metrics: dict[str, float]  # Aggregated (macro-averaged) metrics
    per_query: dict[str, dict[str, float]]  # example id -> per-query metrics


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Complete evaluation report across all modes."""

    modes: list[ModeReport]
    n_answerable: int
    n_skipped_unanswerable: int
    config: dict[str, object]


async def run_eval(
    retriever: Retriever,
    golden: list[GoldenExample],
    *,
    modes: list[str],
    top_k: int = 20,
    config: dict[str, object] | None = None,
) -> EvalReport:
    """Run evaluation over golden examples.

    Args:
        retriever: Retriever instance to evaluate.
        golden: List of golden examples.
        modes: List of retrieval modes to evaluate ("bm25", "dense", "hybrid").
        top_k: Top-k for retrieval (default 20).
        config: Optional config dict to include in report (e.g. fingerprints, model info).

    Returns:
        EvalReport with results.
    """
    if config is None:
        config = {}

    # Filter to answerable examples
    answerable = [ex for ex in golden if ex.type != "unanswerable"]
    unanswerable_count = len(golden) - len(answerable)

    mode_reports = []
    for mode_name in modes:
        mode = RetrievalMode(mode_name)
        per_query: dict[str, dict[str, float]] = {}
        aggregated: dict[str, list[float]] = {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "mrr": [],
            "ndcg@10": [],
        }

        for example in answerable:
            ranking = await retriever.retrieve(example.question, mode=mode, top_k=top_k)
            metrics = evaluate_ranking(example.source_citations, ranking)
            per_query[example.id] = metrics

            # Accumulate for averaging
            for key in aggregated:
                aggregated[key].append(metrics[key])

        # Macro-average
        mode_metrics = {
            key: sum(vals) / len(vals) if vals else 0.0 for key, vals in aggregated.items()
        }

        mode_reports.append(
            ModeReport(
                mode=mode_name,
                metrics=mode_metrics,
                per_query=per_query,
            )
        )

    return EvalReport(
        modes=mode_reports,
        n_answerable=len(answerable),
        n_skipped_unanswerable=unanswerable_count,
        config=config,
    )


def report_markdown(report: EvalReport) -> str:
    """Format report as Markdown table.

    Args:
        report: EvalReport.

    Returns:
        Markdown string with config header and results table.
    """
    lines = []

    # Config header
    lines.append("## Configuration\n")
    for key, val in report.config.items():
        lines.append(f"- {key}: {val}")
    lines.append("")

    # Summary
    lines.append(f"- Answerable examples: {report.n_answerable}")
    lines.append(f"- Unanswerable examples (skipped): {report.n_skipped_unanswerable}")
    lines.append("")

    # Results table
    lines.append("## Results\n")
    lines.append("| Mode | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR | NDCG@10 |")
    lines.append("|------|----------|-----------|-----------|-------------|-----|---------|")

    for mode_report in report.modes:
        m = mode_report.metrics
        recall_5 = f"{m['recall@5']:.4f}"
        recall_10 = f"{m['recall@10']:.4f}"
        recall_20 = f"{m['recall@20']:.4f}"
        precision_5 = f"{m['precision@5']:.4f}"
        mrr = f"{m['mrr']:.4f}"
        ndcg_10 = f"{m['ndcg@10']:.4f}"
        row = (
            f"| {mode_report.mode} | {recall_5} | {recall_10} | {recall_20} | "
            f"{precision_5} | {mrr} | {ndcg_10} |"
        )
        lines.append(row)

    return "\n".join(lines)


def report_json(report: EvalReport) -> dict[str, object]:
    """Convert report to JSON-serializable dict.

    Args:
        report: EvalReport.

    Returns:
        Dictionary suitable for json.dumps.
    """
    return {
        "modes": [
            {
                "mode": mr.mode,
                "metrics": mr.metrics,
                "per_query": mr.per_query,
            }
            for mr in report.modes
        ],
        "n_answerable": report.n_answerable,
        "n_skipped_unanswerable": report.n_skipped_unanswerable,
        "config": report.config,
    }


def write_results(report: EvalReport, results_dir: Path) -> Path:
    """Write report to results directory with timestamp.

    Args:
        report: EvalReport.
        results_dir: Directory to write into.

    Returns:
        Path to written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp: YYYYMMDD-HHMMSSZ (UTC)
    now_utc = datetime.now(timezone.utc)  # noqa: UP017 (requires Python 3.13+)
    timestamp = now_utc.strftime("%Y%m%d-%H%M%SZ")
    filename = f"retrieval-{timestamp}.json"
    filepath = results_dir / filename

    data = report_json(report)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath
