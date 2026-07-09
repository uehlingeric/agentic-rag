#!/usr/bin/env python
"""Re-judge existing generation results with a different judge provider/prompt.

Reads a generation results JSONL file, reconstructs question and cited chunks from
the corpus, and re-judges with a specified provider and prompt version. Replaces
the judge block in each row while preserving generation fields.

Usage:
    uv run python evals/rejudge.py \\
      --rows evals/results/generation-20260709-120000Z/anthropic--hybrid--none.jsonl \\
      [--dataset evals/golden/v2.jsonl] \\
      [--corpus data/corpus/chunks.jsonl] \\
      [--judge-provider google] \\
      [--prompt-version 1] \\
      [--out path.jsonl] \\
      [--limit 10]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from agentic_rag.config import get_settings
from agentic_rag.evals.judge import JudgeParseError, judge_answer, judge_provider_for
from agentic_rag.evals.retrieval import load_golden
from agentic_rag.pipeline.base import CitedChunk
from agentic_rag.providers.registry import get_llm_provider
from agentic_rag.retrieval.base import ChunkRecord

app = typer.Typer()


@app.command()
def main(
    rows: str = typer.Option(..., "--rows", help="Path to generation results JSONL"),
    dataset: str = typer.Option("evals/golden/v2.jsonl", "--dataset", help="Golden dataset path"),
    corpus: str = typer.Option("data/corpus/chunks.jsonl", "--corpus", help="Corpus chunks JSONL"),
    judge_provider: str | None = typer.Option(
        None, "--judge-provider", help="Judge provider (derived from rows if not set)"
    ),
    prompt_version: int | None = typer.Option(
        None, "--prompt-version", help="Judge prompt version"
    ),
    out: str | None = typer.Option(None, "--out", help="Output path (auto-derived if not set)"),
    limit: int | None = typer.Option(None, "--limit", help="Max rows to process"),
) -> None:
    """Re-judge generation results."""
    rows_path = Path(rows)
    if not rows_path.exists():
        typer.secho(f"Error: Rows file not found at {rows}", fg="red", err=True)
        raise typer.Exit(1)

    # Load golden dataset for question lookup
    golden = load_golden(Path(dataset))
    golden_by_id = {ex.id: ex for ex in golden}

    # Load corpus chunks for cited ref resolution
    chunk_index: dict[str, ChunkRecord] = {}
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk_dict = json.loads(line)
                chunk = ChunkRecord.from_json(chunk_dict)
                chunk_index[chunk.chunk_id] = chunk

    # Load rows
    rows_data: list[dict] = []
    with rows_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows_data.append(json.loads(line))
                if limit and len(rows_data) >= limit:
                    break

    # Determine judge provider
    if judge_provider is None and rows_data:
        # Try to derive from the first row's generation provider
        gen_provider = rows_data[0].get("provider")
        if gen_provider:
            settings = get_settings()
            judge_provider = judge_provider_for(gen_provider, settings.judge.providers)
        else:
            typer.secho(
                "Error: Could not derive judge provider from rows; provide --judge-provider",
                fg="red",
                err=True,
            )
            raise typer.Exit(1)
    elif judge_provider is None and not rows_data:
        typer.secho(
            "Error: No rows found and no judge provider specified",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)

    # Determine output path
    if out is None:
        stem = rows_path.stem
        judge_prov_abbr = judge_provider[:3] if judge_provider else "unk"
        prompt_version_str = f"v{prompt_version}" if prompt_version else "latest"
        out = str(rows_path.parent / f"{stem}.rejudge-{judge_prov_abbr}-{prompt_version_str}.jsonl")

    typer.echo(
        f"Re-judging {len(rows_data)} rows with {judge_provider} (prompt {prompt_version})..."
    )
    typer.echo(f"Output: {out}")

    # Get judge LLM
    settings = get_settings()
    judge_llm = get_llm_provider(judge_provider, settings)

    async def _rejudge() -> None:
        """Process rows: skip refusals, re-judge the rest."""
        with open(out, "w", encoding="utf-8") as out_file:
            for i, row in enumerate(rows_data, start=1):
                example_id = row.get("example_id")
                refusal = row.get("refusal", False)

                # Copy row with judge=None first
                output_row = dict(row)

                if refusal:
                    # Skip refusal rows; write through with judge=None
                    output_row["judge"] = None
                    out_file.write(json.dumps(output_row) + "\n")
                    if i % 10 == 0:
                        typer.echo(f"  [{i}] {example_id} (refusal, skipped)", err=True)
                    continue

                # Reconstruct cited chunks
                cited_refs = row.get("cited", [])
                cited_chunks: list[CitedChunk] = []
                for ref in cited_refs:
                    chunk_id = ref.get("chunk_id")
                    if chunk_id in chunk_index:
                        chunk = chunk_index[chunk_id]
                        cited_chunks.append(CitedChunk(marker=ref.get("marker", 0), chunk=chunk))

                # Look up question
                golden_ex = golden_by_id.get(example_id)
                if not golden_ex:
                    typer.secho(
                        f"  Warning: No golden example for {example_id}; skipping",
                        fg="yellow",
                        err=True,
                    )
                    out_file.write(json.dumps(output_row) + "\n")
                    continue

                # Judge
                answer_text = row.get("answer_text", "")
                try:
                    judge_scores = await judge_answer(
                        judge_llm,
                        question=golden_ex.question,
                        answer_text=answer_text,
                        cited=cited_chunks,
                        prompt_version=prompt_version,
                        max_tokens=settings.judge.max_tokens,
                        max_parse_retries=settings.judge.max_parse_retries,
                    )
                    # Update judge block
                    from agentic_rag.evals.records import judge_to_json

                    output_row["judge"] = judge_to_json(judge_scores)
                    typer.echo(f"  [{i}] {example_id} OK", err=True)
                except JudgeParseError as exc:
                    typer.secho(
                        f"  [{i}] {example_id} JUDGE PARSE FAILED: {exc}",
                        fg="yellow",
                        err=True,
                    )
                    output_row["judge"] = None
                except Exception as exc:
                    typer.secho(
                        f"  [{i}] {example_id} ERROR: {exc}",
                        fg="red",
                        err=True,
                    )
                    output_row["judge"] = None

                out_file.write(json.dumps(output_row) + "\n")

    asyncio.run(_rejudge())
    typer.secho(f"Done. Results written to {out}", fg="green")


if __name__ == "__main__":
    app()
