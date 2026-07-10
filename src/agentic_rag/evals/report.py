"""Benchmark report generator for evaluation results.

Assembles a markdown document from:
- Fragment files containing prose (analysis, methodology, etc.)
- JSON result files containing metrics
- Templated table renderers

The report is deterministic and driven by a manifest that specifies the order
and content of sections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Fragment:
    """A prose fragment to include verbatim."""

    path: Path


@dataclass(frozen=True, slots=True)
class Table:
    """A table to render from a results JSON."""

    kind: str  # "retrieval" or "rerank"
    source_path: Path
    params: dict[str, Any]  # e.g., {"baseline_only": True}


# Type alias for manifest entries
ManifestEntry = Fragment | Table

# The committed benchmark run rendered by default; a fresh run overrides it via
# build(generation_summary=...) / build_report.py --generation-summary.
GENERATION_SUMMARY = Path("evals/results/generation-20260709-200512Z/summary.json")

# The committed week-5 vanilla-vs-agentic run; overridden the same way via
# build(agentic_summary=...) / build_report.py --agentic-summary.
AGENTIC_SUMMARY = Path("evals/results/generation-20260710-131027Z/summary.json")


def _round_4dp(value: float) -> str:
    """Format a float to 4 decimal places."""
    return f"{value:.4f}"


def _find_bold_maxima(rows: list[dict[str, float | str]], metric_cols: list[str]) -> set[str]:
    """Find (row_key, metric) pairs for column maxima.

    Returns a set of tuples as strings: "row_key|metric" for cells to bold.
    Metrics where higher is better. Bolding includes ties.
    """
    bold_cells: set[str] = set()

    for col in metric_cols:
        # Extract numeric values for this column
        values = []
        for row in rows:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, str):
                try:
                    values.append((row["mode"], float(val)))
                except ValueError:
                    continue
            else:
                values.append((row["mode"], val))

        if not values:
            continue

        # Find maximum
        max_val = max(v[1] for v in values)

        # Bold all rows that match the max
        for mode_key, val in values:
            if val == max_val:
                bold_cells.add(f"{mode_key}|{col}")

    return bold_cells


def render_retrieval_table(results_path: Path) -> str:
    """Render retrieval table from JSON results.

    Args:
        results_path: Path to retrieval JSON file.

    Returns:
        Markdown table string with headers, separator, and data rows.
        Bolds all column maxima.
    """
    with results_path.open() as f:
        data = json.load(f)

    rows = []
    for mode_report in data["modes"]:
        mode = mode_report["mode"]
        metrics = mode_report["metrics"]
        rows.append(
            {
                "mode": mode,
                "recall@5": _round_4dp(metrics["recall@5"]),
                "recall@10": _round_4dp(metrics["recall@10"]),
                "recall@20": _round_4dp(metrics["recall@20"]),
                "precision@5": _round_4dp(metrics["precision@5"]),
                "mrr": _round_4dp(metrics["mrr"]),
                "ndcg@10": _round_4dp(metrics["ndcg@10"]),
            }
        )

    # Find columns to bold (all are "higher is better" for retrieval)
    metric_cols = ["recall@5", "recall@10", "recall@20", "precision@5", "mrr", "ndcg@10"]
    bold_cells = _find_bold_maxima(rows, metric_cols)

    # Build markdown table
    lines = []
    lines.append("| Mode | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR | NDCG@10 |")
    lines.append("|------|----------|-----------|-----------|-------------|-----|---------|")

    for row in rows:
        mode = row["mode"]
        cells = [mode]
        for col in metric_cols:
            val = row[col]
            if f"{mode}|{col}" in bold_cells:
                val = f"**{val}**"
            cells.append(val)

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _merge_rerank_jsons(llm_path: Path, cross_encoder_path: Path) -> list[dict[str, Any]]:
    """Merge two rerank JSON files into a single mode list.

    Uses baseline rows from the llm file, adds reranked rows from both.
    Preserves original order: per-mode interleaving (bm25, bm25+llm,
    bm25+cross-encoder, dense, dense+llm, dense+cross-encoder, hybrid,
    hybrid+llm, hybrid+cross-encoder).

    Args:
        llm_path: Path to rerank JSON with llm results.
        cross_encoder_path: Path to rerank JSON with cross-encoder results.

    Returns:
        List of mode dicts with mode name and metrics.
    """
    with llm_path.open() as f:
        llm_data = json.load(f)

    with cross_encoder_path.open() as f:
        ce_data = json.load(f)

    modes_by_name: dict[str, Any] = {}

    # Collect all modes
    for mode_report in llm_data["modes"]:
        mode = mode_report["mode"]
        modes_by_name[mode] = mode_report

    for mode_report in ce_data["modes"]:
        mode = mode_report["mode"]
        if "+" in mode:  # Only add reranked rows from cross-encoder
            modes_by_name[mode] = mode_report

    # Order: per-mode (bm25 + variants, then dense + variants, then hybrid + variants)
    ordered_modes = []
    for base in ["bm25", "dense", "hybrid"]:
        if base in modes_by_name:
            ordered_modes.append(modes_by_name[base])
        if f"{base}+llm" in modes_by_name:
            ordered_modes.append(modes_by_name[f"{base}+llm"])
        if f"{base}+cross-encoder" in modes_by_name:
            ordered_modes.append(modes_by_name[f"{base}+cross-encoder"])

    return ordered_modes


def render_rerank_table(results_path: Path, cross_encoder_path: Path | None = None) -> str:
    """Render rerank table from JSON results.

    For LLM and cross-encoder results, shows baseline vs. reranked rows.
    Baseline rows reproduce week-2 retrieval numbers, validating the harness.

    Args:
        results_path: Path to llm rerank JSON file.
        cross_encoder_path: If provided, merge llm and cross-encoder results.

    Returns:
        Markdown table string with headers, separator, and data rows.
        Bolds all column maxima within the full result set.
    """
    if cross_encoder_path is not None:
        # Merge both files
        merged_modes = _merge_rerank_jsons(results_path, cross_encoder_path)
    else:
        # Load just one file
        with results_path.open() as f:
            data = json.load(f)
        merged_modes = data["modes"]

    rows = []
    for mode_report in merged_modes:
        mode = mode_report["mode"]
        metrics = mode_report["metrics"]
        row_dict = {
            "mode": mode,
            "recall@5": _round_4dp(metrics["recall@5"]),
            "recall@10": _round_4dp(metrics["recall@10"]),
            "precision@5": _round_4dp(metrics["precision@5"]),
            "mrr": _round_4dp(metrics["mrr"]),
            "ndcg@10": _round_4dp(metrics["ndcg@10"]),
        }
        rows.append(row_dict)

    # Find columns to bold (all are "higher is better")
    metric_cols = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@10"]
    bold_cells = _find_bold_maxima(rows, metric_cols)

    # Build markdown table (no recall@20 for rerank depth-10 results)
    lines = []
    lines.append("| Mode | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@10 |")
    lines.append("|------|----------|-----------|-------------|-----|---------|")

    for row in rows:
        mode = row["mode"]
        cells = [mode]
        for col in metric_cols:
            val = row[col]
            if f"{mode}|{col}" in bold_cells:
                val = f"**{val}**"
            cells.append(val)

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_agentic_comparison(summary_path: Path) -> str:
    """Render agentic vs. vanilla comparison from summary JSON.

    Input: a summary.json containing BOTH pipelines for >= 1 provider at the
    same mode/rerank. Output markdown:
    1. A headline table, one row per (provider, pipeline) sorted by provider
       then vanilla-first: columns Provider | Pipeline | Faithfulness | Relevance |
       Citation Acc. | False refusal | Refusal correct | Mean revisions |
       Caveat rate | p50 latency (s) | Gen cost ($).
    2. A per-type delta table, one row per (provider, answerable type):
       columns Provider | Type | n | F van | F ag | DF | R van | R ag | DR |
       C van | C ag | DC. Deltas = agentic - vanilla, signed with 2dp.

    Raises ValueError if no agentic config is present.
    """
    with summary_path.open() as f:
        data = json.load(f)

    configs = data["configs"]

    # Filter for agentic configs
    agentic_configs = [c for c in configs if c.get("pipeline") == "agentic"]
    vanilla_configs = [c for c in configs if c.get("pipeline", "vanilla") == "vanilla"]

    if not agentic_configs:
        raise ValueError("Summary contains no agentic config; cannot render comparison")

    lines = []
    lines.append("## Agentic vs. Vanilla Comparison")
    lines.append("")

    # Build headline table: one row per (provider, pipeline)
    headline_rows = []
    providers_in_summary = sorted(set(c["provider"] for c in configs))
    for provider in providers_in_summary:
        # Vanilla row for this provider (if exists)
        vanilla_for_provider = next((c for c in vanilla_configs if c["provider"] == provider), None)
        if vanilla_for_provider is not None:
            headline_rows.append(
                {
                    "provider": provider,
                    "pipeline": "vanilla",
                    "faithfulness": vanilla_for_provider["scores"].get("faithfulness"),
                    "relevance": vanilla_for_provider["scores"].get("relevance"),
                    "citation_accuracy": vanilla_for_provider["scores"].get("citation_accuracy"),
                    "false_refusal": vanilla_for_provider.get("false_refusal_rate"),
                    "refusal_correct": vanilla_for_provider.get("refusal_correct_rate"),
                    "mean_revisions": None,
                    "caveat_rate": None,
                    "p50_latency": vanilla_for_provider.get("latency_s", {}).get("p50"),
                    "gen_cost": vanilla_for_provider.get("gen_cost_usd"),
                }
            )
        # Agentic row for this provider (if exists)
        agentic_for_provider = next((c for c in agentic_configs if c["provider"] == provider), None)
        if agentic_for_provider is not None:
            agent_stats = agentic_for_provider.get("agent") or {}
            headline_rows.append(
                {
                    "provider": provider,
                    "pipeline": "agentic",
                    "faithfulness": agentic_for_provider["scores"].get("faithfulness"),
                    "relevance": agentic_for_provider["scores"].get("relevance"),
                    "citation_accuracy": agentic_for_provider["scores"].get("citation_accuracy"),
                    "false_refusal": agentic_for_provider.get("false_refusal_rate"),
                    "refusal_correct": agentic_for_provider.get("refusal_correct_rate"),
                    "mean_revisions": agent_stats.get("mean_revisions"),
                    "caveat_rate": agent_stats.get("caveat_rate"),
                    "p50_latency": agentic_for_provider.get("latency_s", {}).get("p50"),
                    "gen_cost": agentic_for_provider.get("gen_cost_usd"),
                }
            )

    # Format headline table
    def fmt_score(val: float | None) -> str:
        """Format score to 2dp or —."""
        return f"{val:.2f}" if val is not None else "—"

    def fmt_cost(val: float | None) -> str:
        """Format cost to $X.XX or —."""
        return f"${val:.2f}" if val is not None else "—"

    lines.append(
        "| Provider | Pipeline | Faith. | Relev. | Cit. Acc. | "
        "False refusal | Correct refusal | Mean revisions | Caveat | p50 (s) | Cost ($) |"
    )
    lines.append(
        "|----------|----------|--------|--------|-----------|"
        "-----------------|-----------------|-----------------|--------|---------|---------|"
    )

    for row in headline_rows:
        cells = [
            row["provider"],
            row["pipeline"],
            fmt_score(row["faithfulness"]),
            fmt_score(row["relevance"]),
            fmt_score(row["citation_accuracy"]),
            fmt_score(row["false_refusal"]),
            fmt_score(row["refusal_correct"]),
            fmt_score(row["mean_revisions"]),
            fmt_score(row["caveat_rate"]),
            fmt_score(row["p50_latency"]),
            fmt_cost(row["gen_cost"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Build per-type delta table
    lines.append(
        "### By Type: Faithfulness / Relevance / Citation Accuracy Deltas (agentic - vanilla)"
    )
    lines.append("")

    type_rows = []
    for provider in providers_in_summary:
        vanilla_for_provider = next((c for c in vanilla_configs if c["provider"] == provider), None)
        agentic_for_provider = next((c for c in agentic_configs if c["provider"] == provider), None)
        # Both must exist to compute deltas
        if vanilla_for_provider is None or agentic_for_provider is None:
            continue

        # Get types present in both
        vanilla_types = set(vanilla_for_provider.get("by_type", {}).keys())
        agentic_types = set(agentic_for_provider.get("by_type", {}).keys())
        common_types = vanilla_types & agentic_types

        for ex_type in sorted(common_types):
            # Only include answerable types
            if ex_type == "unanswerable":
                continue
            van_type_data = vanilla_for_provider["by_type"][ex_type]
            ag_type_data = agentic_for_provider["by_type"][ex_type]

            van_f = van_type_data["scores"].get("faithfulness")
            ag_f = ag_type_data["scores"].get("faithfulness")
            delta_f = None if (van_f is None or ag_f is None) else ag_f - van_f

            van_r = van_type_data["scores"].get("relevance")
            ag_r = ag_type_data["scores"].get("relevance")
            delta_r = None if (van_r is None or ag_r is None) else ag_r - van_r

            van_c = van_type_data["scores"].get("citation_accuracy")
            ag_c = ag_type_data["scores"].get("citation_accuracy")
            delta_c = None if (van_c is None or ag_c is None) else ag_c - van_c

            type_rows.append(
                {
                    "provider": provider,
                    "type": ex_type,
                    "n": van_type_data["n"],
                    "f_van": van_f,
                    "f_ag": ag_f,
                    "delta_f": delta_f,
                    "r_van": van_r,
                    "r_ag": ag_r,
                    "delta_r": delta_r,
                    "c_van": van_c,
                    "c_ag": ag_c,
                    "delta_c": delta_c,
                }
            )

    def fmt_delta(val: float | None) -> str:
        """Format delta to +X.XX or -X.XX or —."""
        if val is None:
            return "—"
        if val >= 0:
            return f"+{val:.2f}"
        else:
            return f"{val:.2f}"

    lines.append(
        "| Provider | Type | n | F van | F ag | ΔF | R van | R ag | ΔR | C van | C ag | ΔC |"
    )
    lines.append(
        "|----------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|"
    )

    for row in type_rows:
        cells = [
            row["provider"],
            row["type"],
            str(row["n"]),
            fmt_score(row["f_van"]),
            fmt_score(row["f_ag"]),
            fmt_delta(row["delta_f"]),
            fmt_score(row["r_van"]),
            fmt_score(row["r_ag"]),
            fmt_delta(row["delta_r"]),
            fmt_score(row["c_van"]),
            fmt_score(row["c_ag"]),
            fmt_delta(row["delta_c"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def render_generation_section(summary_path: Path) -> str:
    """Render generation quality and operations section from summary JSON.

    Args:
        summary_path: Path to generation summary JSON.

    Returns:
        Markdown string with heading, judge attribution, quality table,
        ops table, and refusal correctness notes.
    """
    with summary_path.open() as f:
        data = json.load(f)

    run_id = data["run_id"]
    dataset_version = data["dataset_version"]
    n_examples = data["n_examples"]
    configs = data["configs"]

    lines = []

    # Heading and config line
    lines.append("## Generation — provider × retrieval matrix")  # noqa: RUF001
    lines.append("")

    # Synthesis prompt from first config (with validation)
    synthesis_prompts = set()
    for cfg in configs:
        if "synthesis_prompt" in cfg and cfg["synthesis_prompt"] is not None:
            synthesis_prompts.add(cfg["synthesis_prompt"])

    if len(synthesis_prompts) == 1:
        synthesis_prompt_str = next(iter(synthesis_prompts))
    else:
        synthesis_prompt_str = ", ".join(sorted(synthesis_prompts))
    lines.append(
        f"Run `{run_id}`, dataset {dataset_version} ({n_examples} examples), "
        f"synthesis prompt {synthesis_prompt_str}."
    )
    lines.append("")

    # Judge attribution: group by (judge_provider, judge_model, judge_prompt)
    judge_groups: dict[tuple[str, str, str], list[str]] = {}
    for cfg in configs:
        if cfg.get("judge_provider") is None:
            continue
        judge_key = (cfg["judge_provider"], cfg["judge_model"], cfg.get("judge_prompt", ""))
        provider = cfg["provider"]
        if judge_key not in judge_groups:
            judge_groups[judge_key] = []
        judge_groups[judge_key].append(provider)

    if judge_groups:
        for (judge_provider, judge_model, judge_prompt), providers in judge_groups.items():
            sorted_providers = sorted(set(providers))
            providers_str = ", ".join(sorted_providers)
            lines.append(
                f"- {providers_str} answers judged by {judge_provider} "
                f"`{judge_model}` ({judge_prompt})"
            )
        lines.append("")

    # Quality table headers (fields wrapped to manage line length)
    quality_rows = []
    for cfg in configs:
        provider = cfg["provider"]
        mode = cfg["mode"]
        rerank = cfg.get("rerank", "none")
        scores = cfg.get("scores", {})
        faithfulness = scores.get("faithfulness")
        relevance = scores.get("relevance")
        citation_accuracy = scores.get("citation_accuracy")
        refusal_correct = cfg.get("refusal_correct_rate")
        false_refusal = cfg.get("false_refusal_rate")
        n_judged = cfg.get("n_judged", 0)
        n_items = cfg.get("n_items", 0)

        quality_rows.append(
            {
                "provider": provider,
                "mode": mode,
                "rerank": rerank,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "citation_accuracy": citation_accuracy,
                "refusal_correct": refusal_correct,
                "false_refusal": false_refusal,
                "n_judged": n_judged,
                "n_items": n_items,
            }
        )

    # Format quality rows
    def format_score(val: float | None) -> str:
        """Format score to 2 decimals or — if null."""
        return f"{val:.2f}" if val is not None else "—"

    def format_judged(n_judged: int, n_items: int) -> str:
        """Format judged as n_judged/n_items."""
        return f"{n_judged}/{n_items}"

    # Find bold cells for quality table
    # Bold maxima for: faithfulness, relevance, citation_accuracy, refusal_correct
    # Bold MINIMA for: false_refusal
    bold_max_cols = ["faithfulness", "relevance", "citation_accuracy", "refusal_correct"]
    bold_min_cols = ["false_refusal"]

    bold_cells = set()
    for col in bold_max_cols:
        values = [(i, row[col]) for i, row in enumerate(quality_rows) if row[col] is not None]
        if values:
            max_val = max(v[1] for v in values)
            for i, val in values:
                if val == max_val:
                    bold_cells.add((i, col))

    for col in bold_min_cols:
        values = [(i, row[col]) for i, row in enumerate(quality_rows) if row[col] is not None]
        if values:
            min_val = min(v[1] for v in values)
            for i, val in values:
                if val == min_val:
                    bold_cells.add((i, col))

    # Build quality table
    header = (
        "| Provider | Mode | Rerank | Faithfulness | Relevance | "
        "Citation acc. | Refusal correct | False refusal | Judged |"
    )
    lines.append(header)
    lines.append(
        "|----------|------|--------|--------------|-----------|"
        "----------------|-----------------|---------------|--------|"
    )

    quality_cols = [
        "faithfulness",
        "relevance",
        "citation_accuracy",
        "refusal_correct",
        "false_refusal",
    ]
    for i, row in enumerate(quality_rows):
        cells = [row["provider"], row["mode"], row["rerank"]]

        for col in quality_cols:
            val = format_score(row[col])
            if (i, col) in bold_cells:
                val = f"**{val}**"
            cells.append(val)

        cells.append(format_judged(row["n_judged"], row["n_items"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Ops table
    ops_header = (
        "| Provider | Mode | Rerank | Latency mean (s) | p50 | p95 | "
        "Gen tokens in/out | Gen cost | Judge cost |"
    )
    lines.append(ops_header)
    lines.append(
        "|----------|------|--------|------------------|-----|-----|"
        "-------------------|----------|------------|"
    )

    def format_latency(val: float | None) -> str:
        """Format latency to 2 decimals or — if null."""
        return f"{val:.2f}" if val is not None else "—"

    def format_tokens(in_val: int | None, out_val: int | None) -> str:
        """Format tokens as in_val / out_val or — if null."""
        if in_val is None or out_val is None:
            return "—"
        return f"{in_val} / {out_val}"

    def format_cost(val: float | None) -> str:
        """Format cost as $X.XX or — if null."""
        return f"${val:.2f}" if val is not None else "—"

    for cfg in configs:
        provider = cfg["provider"]
        mode = cfg["mode"]
        rerank = cfg.get("rerank", "none")
        latency = cfg.get("latency_s", {})
        latency_mean = latency.get("mean") if latency else None
        latency_p50 = latency.get("p50") if latency else None
        latency_p95 = latency.get("p95") if latency else None
        gen_tokens = cfg.get("gen_tokens", {})
        gen_tokens_in = gen_tokens.get("input") if gen_tokens else None
        gen_tokens_out = gen_tokens.get("output") if gen_tokens else None
        gen_cost = cfg.get("gen_cost_usd")
        judge_cost = cfg.get("judge_cost_usd")

        cells = [
            provider,
            mode,
            rerank,
            format_latency(latency_mean),
            format_latency(latency_p50),
            format_latency(latency_p95),
            format_tokens(gen_tokens_in, gen_tokens_out),
            format_cost(gen_cost),
            format_cost(judge_cost),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # Trailing line and judge failures
    lines.append(
        "Refusal correctness is reported separately from the 1-5 rubric means and is never "
        "averaged into them. Judge failures (n): listed per config when nonzero."
    )

    judge_failures = [
        (f"{cfg['provider']}/{cfg['mode']}/{cfg['rerank']}", cfg.get("n_judge_failures", 0))
        for cfg in configs
        if cfg.get("n_judge_failures", 0) > 0
    ]
    if judge_failures:
        lines.append("")
        for config_slug, n_failures in judge_failures:
            lines.append(f"- {config_slug}: {n_failures}")

    return "\n".join(lines)


def build(
    out_path: Path,
    *,
    generation_summary: Path | None = None,
    agentic_summary: Path | None = None,
) -> str:
    """Build and write the benchmark report.

    Assembles markdown document from fragments and rendered tables.
    Returns the assembled text and writes it to out_path.

    Args:
        out_path: Output path for the markdown document.
        generation_summary: Path to a generation summary JSON; defaults to the
                            pinned committed run (GENERATION_SUMMARY).
        agentic_summary: Path to a vanilla-vs-agentic summary JSON; defaults to
                         the pinned committed run (AGENTIC_SUMMARY).

    Returns:
        The assembled markdown document as a string.
    """
    lines = []

    # Header comment (required)
    lines.append(
        "<!-- Generated by evals/build_report.py — do not edit by hand. "
        "Prose lives in docs/fragments/benchmarks/. -->"
    )
    lines.append("")

    # Title
    lines.append("# Retrieval Benchmarks")
    lines.append("")

    # Manifest: ordered sections with fragments and tables
    manifest: list[ManifestEntry] = [
        Fragment(Path("docs/fragments/benchmarks/01-methodology.md")),
        Fragment(Path("docs/fragments/benchmarks/02-week2-config.md")),
        Table(
            kind="retrieval",
            source_path=Path("evals/results/retrieval-20260704-213430Z.json"),
            params={},
        ),
        Fragment(Path("docs/fragments/benchmarks/03-week2-analysis.md")),
        Fragment(Path("docs/fragments/benchmarks/04-week3-config.md")),
        Table(
            kind="rerank",
            source_path=Path("evals/results/rerank-20260705-011952Z.json"),
            params={
                "cross_encoder_path": Path("evals/results/rerank-20260705-012202Z.json"),
            },
        ),
        Fragment(Path("docs/fragments/benchmarks/05-week3-analysis.md")),
        Table(
            kind="generation",
            source_path=(
                generation_summary if generation_summary is not None else GENERATION_SUMMARY
            ),
            params={},
        ),
        Fragment(Path("docs/fragments/benchmarks/07-week4-analysis.md")),
        Fragment(Path("docs/fragments/benchmarks/08-week5-config.md")),
        Table(
            kind="agentic",
            source_path=agentic_summary if agentic_summary is not None else AGENTIC_SUMMARY,
            params={},
        ),
        Fragment(Path("docs/fragments/benchmarks/09-week5-analysis.md")),
        Fragment(Path("docs/fragments/benchmarks/10-week6-guardrails.md")),
        Fragment(Path("docs/fragments/benchmarks/06-reproduce.md")),
    ]

    # Process manifest
    for entry in manifest:
        if isinstance(entry, Fragment):
            # Read fragment file
            with entry.path.open() as f:
                frag_text = f.read().rstrip()
            lines.append(frag_text)
            lines.append("")
        elif isinstance(entry, Table):
            # Render table based on kind
            if entry.kind == "retrieval":
                table_text = render_retrieval_table(entry.source_path)
            elif entry.kind == "rerank":
                cross_encoder_path = None
                if "cross_encoder_path" in entry.params:
                    cross_encoder_path = entry.params["cross_encoder_path"]
                    if isinstance(cross_encoder_path, str):
                        cross_encoder_path = Path(cross_encoder_path)
                table_text = render_rerank_table(
                    entry.source_path, cross_encoder_path=cross_encoder_path
                )
            elif entry.kind == "generation":
                table_text = render_generation_section(entry.source_path)
            elif entry.kind == "agentic":
                table_text = render_agentic_comparison(entry.source_path)
            else:
                raise ValueError(f"Unknown table kind: {entry.kind}")

            lines.append(table_text)
            lines.append("")

    # Join and clean up trailing whitespace
    document = "\n".join(lines).rstrip()
    document += "\n"  # Ensure file ends with newline

    # Write to output path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(document)

    return document
