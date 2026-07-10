#!/usr/bin/env python
"""Guardrail verification: false-positive rate, overhead, red-team catch rates.

No LLM calls, no cost. Exercises the committed guardrail layer against four
questions the week-6 exit criteria ask honestly:

  1. Input false positives — every clean golden question scanned; a BLOCK on a
     clean question is a false positive (expected: zero).
  2. Output false positives — every non-refusal answer from the committed
     benchmark run scanned; a BLOCK is a false positive (expected: zero).
     Incidental redactions on clean NIST prose are reported too.
  3. Overhead — wall-clock of the full input+output scan+policy path on real
     (question, answer) pairs, reported as p50/p95 (exit bar: <300ms p50).
  4. Red-team catch rate — the injection scanner over evals/redteam/attacks_v1
     .jsonl, per category, with the honest list of documented misses.

Writes evals/results/guardrails-<run-id>/summary.json and prints a markdown
report suitable for docs/guardrails.md.

Usage:
    uv run python evals/run_guardrails.py \\
      [--benchmark-run evals/results/generation-20260710-131027Z] \\
      [--run-id auto-timestamp]
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import typer

from agentic_rag.guardrails.injection import InjectionScanner
from agentic_rag.guardrails.pii import PIIScanner
from agentic_rag.guardrails.policy import apply_policy, default_policy

app = typer.Typer()

_GOLDEN = ["evals/golden/v1.jsonl", "evals/golden/v2.jsonl"]
_REDTEAM = "evals/redteam/attacks_v1.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile in milliseconds (values are seconds)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[rank] * 1000.0


@app.command()
def main(
    benchmark_run: str = typer.Option(
        "evals/results/generation-20260710-131027Z",
        "--benchmark-run",
        help="Committed generation run whose answers seed the output/overhead checks.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Results dir suffix (default: UTC)."),
) -> None:
    """Run guardrail verification and write a results directory + report."""
    policy = default_policy()
    pii = PIIScanner(ner=False)
    injection = InjectionScanner()

    # 1. Input false positives: every clean golden question.
    questions: dict[str, str] = {}
    for rel in _GOLDEN:
        for row in _read_jsonl(Path(rel)):
            questions[str(row["id"])] = str(row["question"])

    input_blocks: list[str] = []
    input_redactions = 0
    for qid, q in questions.items():
        verdict = apply_policy(policy, q, pii.scan(q) + injection.scan(q), direction="input")
        if verdict.blocked:
            input_blocks.append(qid)
        input_redactions += sum(1 for a in verdict.applied if a.action.value == "redact")

    # 2. Output false positives: non-refusal answers from the committed run.
    run_dir = Path(benchmark_run)
    answers: list[tuple[str, str]] = []  # (example_id, answer_text)
    for path in sorted(run_dir.glob("*.jsonl")):
        if ".rejudge" in path.name or path.name == "summary.json":
            continue
        for row in _read_jsonl(path):
            if not row["refusal"] and row["answer_text"]:
                answers.append((str(row["example_id"]), str(row["answer_text"])))

    output_blocks: list[str] = []
    output_redactions = 0
    for eid, text in answers:
        verdict = apply_policy(policy, text, pii.scan(text), direction="output")
        if verdict.blocked:
            output_blocks.append(eid)
        output_redactions += sum(1 for a in verdict.applied if a.action.value == "redact")

    # 3. Overhead: full input+output scan+policy on real (question, answer) pairs.
    overhead_s: list[float] = []
    for eid, text in answers:
        q = questions.get(eid)
        if q is None:
            continue
        start = time.perf_counter()
        apply_policy(policy, q, pii.scan(q) + injection.scan(q), direction="input")
        apply_policy(policy, text, pii.scan(text), direction="output")
        overhead_s.append(time.perf_counter() - start)

    # 4. Red-team catch rate (honest, per category).
    attacks = _read_jsonl(Path(_REDTEAM))
    per_cat: Counter[str] = Counter()
    caught_cat: Counter[str] = Counter()
    misses: list[dict[str, str]] = []
    for atk in attacks:
        category = str(atk["category"])
        caught = any(d.entity == category for d in injection.scan(str(atk["text"])))
        if atk["expect_catch"]:
            per_cat[category] += 1
            caught_cat[category] += int(caught)
        else:
            misses.append({"id": str(atk["id"]), "category": category, "note": str(atk["note"])})

    catch_by_cat = {
        cat: {"caught": caught_cat[cat], "total": per_cat[cat]} for cat in sorted(per_cat)
    }
    total_caught = sum(caught_cat.values())
    total_pos = sum(per_cat.values())

    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark_run": benchmark_run,
        "policy_version": policy.version,
        "input_false_positives": {
            "n_questions": len(questions),
            "blocks": input_blocks,
            "n_blocks": len(input_blocks),
            "incidental_redactions": input_redactions,
        },
        "output_false_positives": {
            "n_answers": len(answers),
            "blocks": output_blocks,
            "n_blocks": len(output_blocks),
            "incidental_redactions": output_redactions,
        },
        "overhead_ms": {
            "n_pairs": len(overhead_s),
            "p50": round(_percentile(overhead_s, 50), 3),
            "p95": round(_percentile(overhead_s, 95), 3),
            "mean": round(statistics.fmean(overhead_s) * 1000, 3) if overhead_s else 0.0,
        },
        "redteam": {
            "n_cases": len(attacks),
            "positives": {"caught": total_caught, "total": total_pos},
            "by_category": catch_by_cat,
            "known_misses": misses,
        },
    }

    suffix = run_id if run_id is not None else datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    out_dir = Path("evals/results") / f"guardrails-{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    typer.echo(_render_markdown(summary))
    typer.echo(f"\nResults written to {out_dir / 'summary.json'}")


def _render_markdown(summary: dict[str, object]) -> str:
    inp = summary["input_false_positives"]
    out = summary["output_false_positives"]
    ov = summary["overhead_ms"]
    rt = summary["redteam"]
    assert isinstance(inp, dict) and isinstance(out, dict) and isinstance(ov, dict)
    assert isinstance(rt, dict)
    lines = [
        "### Guardrail verification",
        "",
        "**False-positive check** (clean traffic must not be blocked):",
        "",
        f"- Input: {inp['n_blocks']} blocks / {inp['n_questions']} golden questions "
        f"({inp['incidental_redactions']} incidental redactions)",
        f"- Output: {out['n_blocks']} blocks / {out['n_answers']} benchmark answers "
        f"({out['incidental_redactions']} incidental redactions)",
        "",
        f"**Overhead** (input+output scan+policy, {ov['n_pairs']} real pairs): "
        f"p50 {ov['p50']} ms, p95 {ov['p95']} ms, mean {ov['mean']} ms",
        "",
        "**Red-team catch rate** (injection scanner, honest):",
        "",
        "| Category | Caught | Total |",
        "| --- | --- | --- |",
    ]
    by_cat = rt["by_category"]
    assert isinstance(by_cat, dict)
    for cat, counts in by_cat.items():
        lines.append(f"| {cat} | {counts['caught']} | {counts['total']} |")
    pos = rt["positives"]
    assert isinstance(pos, dict)
    lines.append(f"| **overall (expect-catch)** | **{pos['caught']}** | **{pos['total']}** |")
    misses = rt["known_misses"]
    assert isinstance(misses, list)
    lines += ["", f"Documented known misses ({len(misses)}): "]
    lines += [f"- `{m['id']}` ({m['category']}): {m['note']}" for m in misses]
    return "\n".join(lines)


if __name__ == "__main__":
    app()
