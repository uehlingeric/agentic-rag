#!/usr/bin/env python
"""Run retrieval evaluation benchmark against the golden dataset.

Usage:
    uv run python evals/run_retrieval.py [--mode bm25 --mode dense --mode hybrid] [--top-k 20]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from agentic_rag.config import get_settings
from agentic_rag.evals.retrieval import (
    load_golden,
    report_markdown,
    run_eval,
    write_results,
)
from agentic_rag.providers.registry import get_embedding_provider
from agentic_rag.retrieval.retriever import Retriever

_DEFAULT_MODES = ["bm25", "dense", "hybrid"]
app = typer.Typer()


@app.command()
def main(
    modes: list[str] = typer.Option(  # noqa: B008
        _DEFAULT_MODES, "--mode", help="Retrieval modes to evaluate"
    ),
    top_k: int = typer.Option(20, "--top-k", help="Top-k for retrieval"),
) -> None:
    """Run retrieval evaluation."""
    settings = get_settings()

    # Load golden dataset
    golden_path = Path("evals/golden/v1.jsonl")
    golden = load_golden(golden_path)

    # Build embedder and retriever
    embedder = get_embedding_provider(settings.embedding.provider, settings)
    retriever = Retriever.load(
        settings.data_dir / "index",
        embedder,
        rrf_k=settings.retrieval.rrf_k,
        candidate_pool=settings.retrieval.candidate_pool,
    )

    # Build config dict
    config = {
        "dataset_path": str(golden_path),
        "n_questions": len(golden),
        "n_answerable": sum(1 for ex in golden if ex.type != "unanswerable"),
        "embedding_provider": settings.embedding.provider,
        "embedding_model": settings.embedding.model,
        "rrf_k": settings.retrieval.rrf_k,
        "candidate_pool": settings.retrieval.candidate_pool,
        "top_k": top_k,
    }

    # Run evaluation
    report = asyncio.run(run_eval(retriever, golden, modes=modes, top_k=top_k, config=config))

    # Print markdown report to stdout
    print(report_markdown(report))

    # Write JSON results to evals/results/
    results_dir = Path("evals/results")
    filepath = write_results(report, results_dir)
    print(f"\nResults written to: {filepath}", file=sys.stderr)


if __name__ == "__main__":
    app()
