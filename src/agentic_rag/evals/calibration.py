"""Judge calibration: inter-rater agreement and labeling-sheet generation.

Quadratic-weighted Cohen's kappa measures ordinal agreement between two raters.
The labeling sheet renders examples and provides a JSON template for manual labeling.
Stratified selection ensures diversity across types and configurations.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.evals.judge import DIMENSIONS
from agentic_rag.evals.retrieval import GoldenExample
from agentic_rag.retrieval.base import ChunkRecord


def quadratic_weighted_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """Compute quadratic-weighted Cohen's kappa for ordinal agreement.

    Measures agreement between two raters on a scale [min_rating, max_rating].
    Uses quadratic weights w_ij = (i-j)^2 / (k-1)^2 where k = max_rating - min_rating + 1.

    kappa = 1 - (observed disagreement) / (expected disagreement)

    Expected disagreement is computed from the outer product of rater marginals.

    Degenerate case: If both raters assign identical constant ratings (all items
    the same), kappa is undefined in the standard formula (expected disagreement
    is zero). We return 1.0 in this case, as it indicates perfect agreement.

    Args:
        rater_a: Sequence of ratings from rater A.
        rater_b: Sequence of ratings from rater B.
        min_rating: Minimum rating on the scale (default 1).
        max_rating: Maximum rating on the scale (default 5).

    Returns:
        Kappa coefficient in [-1, 1], where 1.0 is perfect agreement.

    Raises:
        ValueError: If lengths differ, are zero, or any rating is outside [min, max].
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"rater sequences must have equal length: {len(rater_a)} != {len(rater_b)}"
        )
    if len(rater_a) == 0:
        raise ValueError("rater sequences must be non-empty")

    # Validate ratings
    for rating in rater_a:
        if not isinstance(rating, int) or not (min_rating <= rating <= max_rating):
            raise ValueError(f"rater_a rating {rating} outside [{min_rating}, {max_rating}]")
    for rating in rater_b:
        if not isinstance(rating, int) or not (min_rating <= rating <= max_rating):
            raise ValueError(f"rater_b rating {rating} outside [{min_rating}, {max_rating}]")

    n = len(rater_a)
    k = max_rating - min_rating + 1

    # Map to 0-indexed scale for weight computation
    a_idx = [r - min_rating for r in rater_a]
    b_idx = [r - min_rating for r in rater_b]

    # Observed disagreement: sum of weighted differences
    weight_scale = (k - 1) ** 2
    observed = sum((a_idx[i] - b_idx[i]) ** 2 for i in range(n)) / weight_scale / n

    # Expected disagreement: marginal-based outer product
    count_a = Counter(a_idx)
    count_b = Counter(b_idx)

    expected = 0.0
    for i in range(k):
        for j in range(k):
            p_a = count_a[i] / n
            p_b = count_b[j] / n
            weight = (i - j) ** 2 / weight_scale
            expected += p_a * p_b * weight

    # Degenerate case: if expected == 0, both raters are constant on the same value
    if expected == 0.0:
        return 1.0

    return 1.0 - (observed / expected)


@dataclass(frozen=True, slots=True)
class CalibrationLabel:
    """One manual label row for calibration."""

    example_id: str
    labeler: str
    faithfulness: int
    relevance: int
    citation_accuracy: int
    notes: str | None = None


def load_labels(path: Path) -> list[CalibrationLabel]:
    """Load calibration labels from JSONL.

    Args:
        path: Path to JSONL file with rows of CalibrationLabel.

    Returns:
        List of CalibrationLabel, preserving file order.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If any row has invalid schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found at {path}")

    labels = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                label = CalibrationLabel(
                    example_id=row["example_id"],
                    labeler=row["labeler"],
                    faithfulness=int(row["faithfulness"]),
                    relevance=int(row["relevance"]),
                    citation_accuracy=int(row["citation_accuracy"]),
                    notes=row.get("notes"),
                )
                labels.append(label)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Line {line_no}: {e}") from e

    return labels


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Per-dimension inter-rater agreement metrics."""

    n: int
    dimensions: dict[
        str, dict[str, object]
    ]  # dim -> {kappa, exact_match_rate, mad, disagreement_ids}

    def table(self) -> str:
        """Render agreement metrics as a markdown table."""
        lines = ["| Dimension | n | Kappa | Exact Match | MAD |"]
        lines.append("|-----------|---|-------|-------------|-----|")

        for dim in DIMENSIONS:
            if dim not in self.dimensions:
                continue
            metrics = self.dimensions[dim]
            kappa = metrics["kappa"]
            exact = metrics["exact_match_rate"]
            mad = metrics["mad"]
            row = f"| {dim} | {self.n} | {kappa:.3f} | {exact:.1%} | {mad:.2f} |"
            lines.append(row)

        return "\n".join(lines)


def agreement(
    labels: Sequence[CalibrationLabel], judged_rows: Sequence[dict[str, object]]
) -> AgreementReport:
    """Compute inter-rater agreement on shared examples.

    Joins labels and rows on example_id. Only rows with non-null judge block are included.
    Per dimension: compute quadratic-weighted kappa, exact-match rate, mean absolute
    difference, and list of examples with |diff| >= 2.

    Args:
        labels: Manual calibration labels.
        judged_rows: Generation records with judge blocks.

    Returns:
        AgreementReport with per-dimension metrics and disagreement audit trail.
    """
    # Index rows by example_id, filtering to those with judge blocks
    row_by_id: dict[str, dict[str, object]] = {}
    for row in judged_rows:
        if row.get("judge") is not None:
            ex_id = row.get("example_id")
            if isinstance(ex_id, str):
                if ex_id in row_by_id:
                    raise ValueError(
                        f"duplicate judged row for {ex_id!r}: labels bind to one specific "
                        "answer, pass a single coherent row set"
                    )
                row_by_id[ex_id] = row

    # Collect per-dimension ratings and disagreements
    dim_labels: dict[str, list[int]] = {}
    dim_rows: dict[str, list[int]] = {}
    dim_disagreements: dict[str, set[str]] = {}
    for dim in DIMENSIONS:
        dim_labels[dim] = []
        dim_rows[dim] = []
        dim_disagreements[dim] = set()

    for label in labels:
        if label.example_id not in row_by_id:
            continue

        row = row_by_id[label.example_id]
        judge_obj = row.get("judge")
        if not isinstance(judge_obj, dict):
            continue

        for dim in DIMENSIONS:
            label_val = getattr(label, dim)
            dim_obj = judge_obj.get(dim)
            if not isinstance(dim_obj, dict):
                continue
            judge_val = dim_obj.get("score")

            if not isinstance(judge_val, int):
                continue

            dim_labels[dim].append(label_val)
            dim_rows[dim].append(judge_val)

            if abs(label_val - judge_val) >= 2:
                dim_disagreements[dim].add(label.example_id)

    # Compute metrics per dimension
    dimensions: dict[str, dict[str, object]] = {}
    n_shared = 0

    for dim in DIMENSIONS:
        labels_seq = dim_labels[dim]
        rows_seq = dim_rows[dim]

        if not labels_seq:
            continue

        n_shared = max(n_shared, len(labels_seq))

        kappa = quadratic_weighted_kappa(labels_seq, rows_seq)
        exact_match = sum(
            1
            for label_val, row_val in zip(labels_seq, rows_seq, strict=True)
            if label_val == row_val
        ) / len(labels_seq)
        mad = sum(
            abs(label_val - row_val)
            for label_val, row_val in zip(labels_seq, rows_seq, strict=True)
        ) / len(labels_seq)
        disagreements = sorted(dim_disagreements[dim])

        dimensions[dim] = {
            "kappa": kappa,
            "exact_match_rate": exact_match,
            "mad": mad,
            "disagreement_ids": disagreements,
        }

    return AgreementReport(n=n_shared, dimensions=dimensions)


def labeling_sheet(
    rows: Sequence[dict[str, object]],
    golden_by_id: Mapping[str, GoldenExample],
    chunks_by_id: Mapping[str, ChunkRecord],
) -> str:
    """Generate a markdown labeling sheet for manual calibration.

    For each row (sorted by example_id):
    - Heading with example_id, provider, mode
    - Question text from golden
    - Answer text
    - Cited excerpts with [marker] doc §section — heading format
    - JSON label template (skipped for refusals)

    Args:
        rows: Generation records.
        golden_by_id: Mapping of example_id to GoldenExample.
        chunks_by_id: Mapping of chunk_id to ChunkRecord.

    Returns:
        Markdown string suitable for writing to a file.
    """
    lines: list[str] = []

    # Sort rows by example_id for reproducibility
    def get_example_id(r: dict[str, object]) -> str:
        ex_id = r.get("example_id")
        return ex_id if isinstance(ex_id, str) else ""

    sorted_rows = sorted(rows, key=get_example_id)

    for row in sorted_rows:
        ex_id = row.get("example_id")
        if not isinstance(ex_id, str):
            continue

        provider_obj = row.get("provider", "unknown")
        provider = provider_obj if isinstance(provider_obj, str) else "unknown"
        mode_obj = row.get("mode", "unknown")
        mode = mode_obj if isinstance(mode_obj, str) else "unknown"
        answer_text_obj = row.get("answer_text", "")
        answer_text = answer_text_obj if isinstance(answer_text_obj, str) else ""
        refusal_obj = row.get("refusal", False)
        refusal = refusal_obj if isinstance(refusal_obj, bool) else False
        cited_obj = row.get("cited", [])
        cited = cited_obj if isinstance(cited_obj, list) else []

        golden = golden_by_id.get(ex_id)
        if not golden:
            continue

        # Heading
        lines.append(f"## {ex_id}")
        lines.append(f"**Provider:** {provider} | **Mode:** {mode}\n")

        # Question
        lines.append("### Question")
        lines.append(f"{golden.question}\n")

        # Answer
        lines.append("### Answer")
        lines.append(f"{answer_text}\n")

        if refusal:
            lines.append(
                "⚠️ **Refusal:** This answer is a refusal and is excluded from rubric labeling.\n"
            )
        else:
            # Cited excerpts
            if cited:
                lines.append("### Cited Excerpts")
                for cite in cited:
                    if not isinstance(cite, dict):
                        continue
                    marker = cite.get("marker", "?")
                    chunk_id = cite.get("chunk_id")
                    if not isinstance(chunk_id, str):
                        continue
                    chunk = chunks_by_id.get(chunk_id)
                    if chunk:
                        heading = chunk.heading or "(no heading)"
                        lines.append(f"[{marker}] {chunk.doc_id} §{chunk.section_id} — {heading}")
                        lines.append(f"```\n{chunk.text}\n```\n")
                lines.append("")

            # JSON label template
            template = {
                "example_id": ex_id,
                "labeler": "",
                "faithfulness": None,
                "relevance": None,
                "citation_accuracy": None,
                "notes": "",
            }
            lines.append("### Label")
            lines.append("```json")
            lines.append(json.dumps(template, indent=2))
            lines.append("```\n")

    return "\n".join(lines)


def select_calibration_items(
    rows: Sequence[dict[str, object]],
    golden_by_id: Mapping[str, GoldenExample],
    n: int = 20,
    *,
    seed: int = 13,
) -> list[dict[str, object]]:
    """Deterministically select stratified calibration items.

    A calibration item is one specific answer row, and labels are keyed by
    example_id — so the selection never picks the same example_id twice, even
    when the input mixes several configs that answered the same question.
    Refusal rows are excluded entirely: they carry no rubric signal (refusal
    correctness is computed mechanically) and would burn label slots.

    Strategy: bucket non-refusal rows by golden question type, shuffle each
    bucket with the seeded rng, then drain buckets round-robin so every type
    present is covered before any type is drawn from twice. Deterministic for
    a given (rows, seed).

    Args:
        rows: Generation records, possibly spanning several configs.
        golden_by_id: Mapping of example_id to GoldenExample.
        n: Target number of items to select (default 20).
        seed: Random seed for determinism (default 13).

    Returns:
        Up to n selected rows with unique example_ids, in input order.
    """
    rng = random.Random(seed)

    # Bucket labelable rows by golden type
    by_type: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or example_id not in golden_by_id:
            continue
        if row.get("refusal", False):
            continue
        by_type.setdefault(golden_by_id[example_id].type, []).append(row)

    for bucket in by_type.values():
        rng.shuffle(bucket)

    # Round-robin across types until n rows or all buckets are dry
    selected: list[dict[str, object]] = []
    chosen_ids: set[str] = set()
    types = sorted(by_type)
    drained = False
    while len(selected) < n and not drained:
        drained = True
        for typ in types:
            if len(selected) >= n:
                break
            bucket = by_type[typ]
            while bucket:
                row = bucket.pop()
                row_id = row["example_id"]
                if isinstance(row_id, str) and row_id not in chosen_ids:
                    selected.append(row)
                    chosen_ids.add(row_id)
                    drained = False
                    break

    # Return in input order so the sheet reads predictably
    positions = {id(r): i for i, r in enumerate(rows)}
    selected.sort(key=lambda r: positions[id(r)])
    return selected
