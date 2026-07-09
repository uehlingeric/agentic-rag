#!/usr/bin/env python
"""Run generation evaluation: answer quality and citation accuracy via LLM-as-judge.

Usage:
    uv run python evals/run_generation.py \\
      --provider ollama --provider anthropic \\
      --mode bm25 --mode dense \\
      --rerank none --rerank llm \\
      [--dataset evals/golden/v2.jsonl] \\
      [--run-id auto-timestamp] \\
      [--concurrency 4] \\
      [--judge | --no-judge] \\
      [--estimate-only] \\
      [--yes]
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from agentic_rag.config import get_settings
from agentic_rag.evals.generation import RunConfig, estimate_cost, eval_set, run_config, summarize
from agentic_rag.evals.retrieval import load_golden

_DEFAULT_PROVIDERS = ["ollama"]
_DEFAULT_MODES = ["bm25", "dense", "hybrid"]
_DEFAULT_RERANKS = ["none"]
_DEFAULT_PIPELINES = ["vanilla"]

app = typer.Typer()


def _format_cost(cost: float | None) -> str:
    """Format cost for display."""
    if cost is None:
        return "?"
    return f"${cost:.2f}"


@app.command()
def main(
    providers: list[str] = typer.Option(  # noqa: B008
        _DEFAULT_PROVIDERS, "--provider", help="LLM providers to evaluate"
    ),
    modes: list[str] = typer.Option(  # noqa: B008
        _DEFAULT_MODES, "--mode", help="Retrieval modes to evaluate"
    ),
    reranks: list[str] = typer.Option(  # noqa: B008
        _DEFAULT_RERANKS, "--rerank", help="Rerank modes to evaluate"
    ),
    pipelines: list[str] = typer.Option(  # noqa: B008
        _DEFAULT_PIPELINES, "--pipeline", help="Pipeline variants: vanilla|agentic"
    ),
    dataset: str = typer.Option("evals/golden/v2.jsonl", "--dataset", help="Golden dataset path"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run identifier; auto-generated if not provided"
    ),
    concurrency: int = typer.Option(4, "--concurrency", help="Concurrent synthesis/judge calls"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Enable LLM-as-judge scoring"),
    estimate_only: bool = typer.Option(
        False, "--estimate-only", help="Print cost estimate and exit"
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip cost confirmation prompt"),
) -> None:
    """Run generation evaluation over golden examples."""
    # Load golden dataset
    try:
        golden_full = load_golden(Path(dataset))
    except FileNotFoundError as exc:
        typer.secho(f"Error: Golden dataset not found at {dataset}", fg="red", err=True)
        raise typer.Exit(1) from exc
    except ValueError as e:
        typer.secho(f"Error parsing dataset: {e}", fg="red", err=True)
        raise typer.Exit(1) from e

    # Drop held-out items for generation eval
    golden = eval_set(golden_full)
    n_held_out = len(golden_full) - len(golden)
    if n_held_out > 0:
        typer.echo(
            f"Excluded {n_held_out} held-out items (planner few-shots); evaluating {len(golden)}."
        )

    # Build config matrix
    configs = [
        RunConfig(provider=p, mode=m, rerank=r, pipeline=pl)
        for p in providers
        for m in modes
        for r in reranks
        for pl in pipelines
    ]

    # Load settings for cost estimation
    settings = get_settings()

    # Estimate costs
    cost_estimates = estimate_cost(configs, len(golden), settings)

    # Print cost table
    typer.echo("\nEstimated costs per config:")
    typer.echo("=" * 100)
    header = f"{'Provider':<12} {'Mode':<8} {'Rerank':<15} {'Pipeline':<10} {'Cost':<12}"
    typer.echo(header)
    typer.echo("-" * 100)

    total_cost = 0.0
    has_unpriceable = False
    for cfg, cost in cost_estimates:
        cost_str = _format_cost(cost)
        if cost is None:
            has_unpriceable = True
        else:
            total_cost += cost
        typer.echo(
            f"{cfg.provider:<12} {cfg.mode:<8} {cfg.rerank:<15} {cfg.pipeline:<10} {cost_str:<12}"
        )

    typer.echo("-" * 100)
    total_str = _format_cost(total_cost) if not has_unpriceable else "? (some configs unpriceable)"
    typer.echo(f"{'TOTAL ESTIMATED COST:':<47} {total_str}")
    if has_unpriceable:
        typer.echo("(Unpriceable: ? denotes unknown model prices; assumes 0 cost)")
    typer.echo()

    if estimate_only:
        raise typer.Exit(0)

    # Confirm above $5 — and always when any config is unpriceable (unknown ≠ free)
    if (
        (total_cost > 5.0 or has_unpriceable)
        and not yes
        and not typer.confirm(f"Proceed with an estimated {total_str} in evaluations?")
    ):
        typer.secho("Cancelled.", fg="yellow")
        raise typer.Exit(0)

    # Generate run-id if needed
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")

    # Create results directory
    results_dir = Path("evals/results") / f"generation-{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\nRunning {len(configs)} configs against {len(golden)} golden examples...")
    typer.echo(f"Results: {results_dir}")
    typer.echo()

    # Run configs sequentially (items concurrent within each)
    async def _run() -> None:
        for i, cfg in enumerate(configs, start=1):
            out_path = results_dir / f"{cfg.slug()}.jsonl"
            typer.echo(f"[{i}/{len(configs)}] {cfg.slug()}...", err=True)
            await run_config(
                cfg,
                golden,
                settings,
                out_path,
                dataset_version=Path(dataset).stem,
                concurrency=concurrency,
                do_judge=judge,
            )

    asyncio.run(_run())

    # Summarize results
    typer.echo("\nAggregating results...", err=True)
    summary = summarize(results_dir, golden)

    # Write summary.json
    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    typer.echo("\n" + "=" * 120)
    typer.echo("GENERATION EVALUATION RESULTS")
    typer.echo("=" * 120)
    typer.echo(f"Run ID: {summary['run_id']}")
    typer.echo(f"Dataset: {dataset} ({summary['n_examples']} examples)")
    typer.echo()

    # Print per-config table
    typer.echo("Config Details:")
    typer.echo("-" * 120)
    for cfg_summary in summary["configs"]:
        typer.echo()
        provider = cfg_summary["provider"]
        mode = cfg_summary["mode"]
        rerank = cfg_summary["rerank"]
        pipeline = cfg_summary.get("pipeline", "vanilla")
        typer.echo(f"  {provider:<12} | {mode:<8} | {rerank:<15} | {pipeline:<10}")
        typer.echo(f"    Model: {cfg_summary['model']}")
        n_items = cfg_summary["n_items"]
        n_judged = cfg_summary["n_judged"]
        n_refusals = cfg_summary["n_refusals"]
        n_judge_failures = cfg_summary["n_judge_failures"]
        typer.echo(
            f"    Items: {n_items} | Judged: {n_judged} | "
            f"Refusals: {n_refusals} | Judge failures: {n_judge_failures}"
        )

        if cfg_summary["n_judged"] > 0:
            scores = cfg_summary["scores"]
            faith = scores["faithfulness"]
            relev = scores["relevance"]
            cite = scores["citation_accuracy"]
            typer.echo(
                f"    Scores (mean): Faithfulness {faith} | "
                f"Relevance {relev} | Citation Accuracy {cite}"
            )

        if cfg_summary["refusal_correct_rate"] is not None:
            rcr = cfg_summary["refusal_correct_rate"]
            typer.echo(f"    Refusal Correctness: {rcr:.4f}")

        if cfg_summary["false_refusal_rate"] is not None:
            frr = cfg_summary["false_refusal_rate"]
            typer.echo(f"    False Refusal Rate: {frr:.4f}")

        latency = cfg_summary["latency_s"]
        if latency:
            mean_l = latency["mean"]
            p50_l = latency["p50"]
            p95_l = latency["p95"]
            typer.echo(f"    Latency (s): mean {mean_l} | p50 {p50_l} | p95 {p95_l}")

        tokens = cfg_summary["gen_tokens"]
        typer.echo(
            f"    Generation: {tokens['input']} input tokens | {tokens['output']} output tokens"
        )
        gen_cost = _format_cost(cfg_summary["gen_cost_usd"])
        judge_cost = _format_cost(cfg_summary["judge_cost_usd"])
        typer.echo(f"    Cost: {gen_cost} generation | {judge_cost} judging")

        # Agent stats for agentic configs
        if pipeline == "agentic" and cfg_summary.get("agent") is not None:
            agent = cfg_summary["agent"]
            typer.echo(
                f"    Agent: multi-hop rate {agent['multi_hop_rate']:.4f} | "
                f"mean revisions {agent['mean_revisions']:.4f} | "
                f"caveat rate {agent['caveat_rate']:.4f}"
            )

    typer.echo()
    typer.echo(f"Summary written to: {summary_path}")
    typer.echo()


if __name__ == "__main__":
    app()
