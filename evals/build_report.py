#!/usr/bin/env python3
"""Build benchmark report from results JSONs and fragments.

Usage:
    uv run python evals/build_report.py                  # Regenerate docs/benchmarks.md
    uv run python evals/build_report.py --out output.md  # Write to custom path
"""

from __future__ import annotations

from pathlib import Path

import typer

from agentic_rag.evals.report import build

app = typer.Typer()


@app.command()
def main(
    out: str = typer.Option(
        "docs/benchmarks.md",
        "--out",
        help="Output path for the markdown document.",
    ),
    generation_summary: str | None = typer.Option(
        None,
        "--generation-summary",
        help="Optional path to generation results summary JSON.",
    ),
    agentic_summary: str | None = typer.Option(
        None,
        "--agentic-summary",
        help="Optional path to a vanilla-vs-agentic results summary JSON.",
    ),
) -> None:
    """Build and write the benchmark report.

    Assembles markdown document from fragments and rendered tables, writing to --out.
    If --generation-summary / --agentic-summary are provided, they override the
    pinned committed runs.

    Args:
        out: Output markdown file path.
        generation_summary: Optional generation results JSON.
        agentic_summary: Optional vanilla-vs-agentic results JSON.
    """
    out_path = Path(out)
    gen_summary = Path(generation_summary) if generation_summary else None
    ag_summary = Path(agentic_summary) if agentic_summary else None
    build(out_path, generation_summary=gen_summary, agentic_summary=ag_summary)
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
