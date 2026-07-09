#!/usr/bin/env python
"""Judge calibration: inter-rater agreement and labeling sheet generation.

Usage:
    uv run python evals/run_calibration.py sheet --rows <path> [--rows <path> ...]
    uv run python evals/run_calibration.py agreement --labels <path> --rows <path>
        [--rows <path> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from agentic_rag.evals.calibration import (
    agreement,
    labeling_sheet,
    load_labels,
    select_calibration_items,
)
from agentic_rag.evals.retrieval import load_golden
from agentic_rag.retrieval.base import load_chunks

app = typer.Typer()


def load_rows(paths: list[str]) -> list[dict[str, object]]:
    """Load generation records from JSONL files."""
    rows: list[dict[str, object]] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Rows file not found: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


@app.command()
def sheet(
    rows: list[str] = typer.Option(  # noqa: B008
        ..., "--rows", help="JSONL path(s) with generation records"
    ),
    dataset: str = typer.Option(
        "evals/golden/v2.jsonl", "--dataset", help="Path to golden dataset"
    ),
    corpus: str = typer.Option("data/corpus/chunks.jsonl", "--corpus", help="Path to chunk corpus"),
    n: int = typer.Option(20, "--n", help="Number of items to select"),
    seed: int = typer.Option(13, "--seed", help="Random seed for stratification"),
    out: str = typer.Option(
        "evals/calibration/sheet.md", "--out", help="Output path for labeling sheet"
    ),
) -> None:
    """Generate a labeling sheet for calibration."""
    # Load data
    all_rows = load_rows(rows)
    golden = load_golden(Path(dataset))
    golden_by_id = {ex.id: ex for ex in golden}
    chunks = load_chunks(Path(corpus))
    chunks_by_id = {ch.chunk_id: ch for ch in chunks}

    # Select stratified items
    selected = select_calibration_items(all_rows, golden_by_id, n=n, seed=seed)

    # Generate sheet
    markdown = labeling_sheet(selected, golden_by_id, chunks_by_id)

    # Write output
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    # Write selected rows as companion file
    selected_path = out_path.parent / "selected.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    print(f"Sheet written to: {out_path}", file=sys.stderr)
    print(f"Selected rows written to: {selected_path}", file=sys.stderr)


@app.command()
def agreement_cmd(
    labels: str = typer.Option(
        "evals/calibration/labels.jsonl",
        "--labels",
        help="Path to manual calibration labels",
    ),
    rows: list[str] = typer.Option(  # noqa: B008
        ..., "--rows", help="JSONL path(s) with judged generation records"
    ),
) -> None:
    """Compute inter-rater agreement metrics."""
    # Load data
    calib_labels = load_labels(Path(labels))
    judged_rows = load_rows(rows)

    # Compute agreement
    report = agreement(calib_labels, judged_rows)

    # Print table
    print(report.table())
    print("")

    # Print disagreement audit trail
    all_disagreements: dict[str, list[str]] = {}
    for dim, metrics in report.dimensions.items():
        ids = metrics["disagreement_ids"]
        if ids:
            all_disagreements[dim] = ids

    if all_disagreements:
        print("## Disagreement Audit (|diff| >= 2)")
        print("")
        for dim in sorted(all_disagreements.keys()):
            ids = all_disagreements[dim]
            print(f"### {dim}")
            for ex_id in ids:
                print(f"- {ex_id}")
            print("")


if __name__ == "__main__":
    app()
