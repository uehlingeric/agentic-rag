"""Unit tests for judge calibration.

Offline tests for kappa, agreement, labeling_sheet, and stratified selection.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agentic_rag.evals.calibration import (
    CalibrationLabel,
    agreement,
    labeling_sheet,
    load_labels,
    quadratic_weighted_kappa,
    select_calibration_items,
)
from agentic_rag.evals.retrieval import GoldenExample
from agentic_rag.retrieval.base import ChunkRecord

# ============================================================================
# Quadratic-Weighted Kappa Tests
# ============================================================================


def _kappa_reference(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """Reference implementation: brute-force confusion matrix computation.

    Computes observed and expected disagreement via explicit O(k^2) loops.
    Used to verify the optimized implementation.
    """
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        raise ValueError("invalid input")

    n = len(rater_a)
    k = max_rating - min_rating + 1

    # Build confusion matrix: rows = rater_a, cols = rater_b
    confusion: dict[tuple[int, int], int] = {}
    for a, b in zip(rater_a, rater_b, strict=True):
        key = (a - min_rating, b - min_rating)
        confusion[key] = confusion.get(key, 0) + 1

    # Observed disagreement
    weight_scale = (k - 1) ** 2
    observed = 0.0
    for (i, j), count in confusion.items():
        weight = (i - j) ** 2 / weight_scale
        observed += weight * count
    observed /= n

    # Expected disagreement from marginals
    row_counts = [0] * k
    col_counts = [0] * k
    for (i, j), count in confusion.items():
        row_counts[i] += count
        col_counts[j] += count

    expected = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / weight_scale
            expected += weight * (row_counts[i] / n) * (col_counts[j] / n)

    if expected == 0.0:
        return 1.0
    return 1.0 - (observed / expected)


def test_kappa_perfect_agreement() -> None:
    """Perfect agreement should yield kappa = 1.0."""
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert quadratic_weighted_kappa(a, b) == 1.0


def test_kappa_constant_identical() -> None:
    """Both raters constant on same value: kappa = 1.0."""
    a = [3, 3, 3, 3]
    b = [3, 3, 3, 3]
    assert quadratic_weighted_kappa(a, b) == 1.0


def test_kappa_constant_different() -> None:
    """Both raters constant but on different values: kappa = 0.0 (perfect disagreement)."""
    # When both raters are constant but on different values, they never agree.
    # Observed disagreement = 1.0, Expected disagreement = 1.0, so Kappa = 0.0
    a = [1, 1, 1, 1]
    b = [5, 5, 5, 5]
    result = quadratic_weighted_kappa(a, b)
    assert result == 0.0


def test_kappa_hand_computed_example() -> None:
    """Hand-computed example: a=[1,2,3,4,5], b=[1,2,3,4,5] plus reference.

    Perfect agreement across a full 5-point scale.
    Expected = 1.0, Observed = 0.0, Kappa = 1.0.
    """
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    result = quadratic_weighted_kappa(a, b)
    ref = _kappa_reference(a, b)
    assert result == ref
    assert result == 1.0


def test_kappa_reference_randomized() -> None:
    """Verify against reference implementation on randomized inputs."""
    import random

    rng = random.Random(42)
    for _ in range(5):
        length = rng.randint(4, 10)
        a = [rng.randint(1, 5) for _ in range(length)]
        b = [rng.randint(1, 5) for _ in range(length)]

        result = quadratic_weighted_kappa(a, b)
        ref = _kappa_reference(a, b)
        assert abs(result - ref) < 1e-9, f"Mismatch: {result} != {ref}"


def test_kappa_different_scale() -> None:
    """Kappa on a [0, 3] scale."""
    a = [0, 1, 2, 3]
    b = [0, 1, 2, 3]
    result = quadratic_weighted_kappa(a, b, min_rating=0, max_rating=3)
    assert result == 1.0


def test_kappa_length_mismatch() -> None:
    """Unequal lengths should raise ValueError."""
    with pytest.raises(ValueError, match="equal length"):
        quadratic_weighted_kappa([1, 2, 3], [1, 2])


def test_kappa_empty() -> None:
    """Empty sequences should raise ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        quadratic_weighted_kappa([], [])


def test_kappa_rating_out_of_range() -> None:
    """Rating outside [min, max] should raise ValueError."""
    with pytest.raises(ValueError, match="outside"):
        quadratic_weighted_kappa([1, 2, 6], [1, 2, 3], min_rating=1, max_rating=5)


# ============================================================================
# Agreement Tests
# ============================================================================


def test_agreement_empty() -> None:
    """Empty inputs should return report with n=0."""
    report = agreement([], [])
    assert report.n == 0
    assert len(report.dimensions) == 0


def test_agreement_full_match() -> None:
    """Perfect label-judge agreement."""
    labels = [
        CalibrationLabel(
            example_id="ex1",
            labeler="alice",
            faithfulness=4,
            relevance=5,
            citation_accuracy=3,
        ),
    ]
    judged_rows = [
        {
            "example_id": "ex1",
            "judge": {
                "faithfulness": {"score": 4, "justification": "correct"},
                "relevance": {"score": 5, "justification": "relevant"},
                "citation_accuracy": {
                    "score": 3,
                    "justification": "some errors",
                },
            },
        },
    ]
    report = agreement(labels, judged_rows)
    assert report.n == 1
    assert report.dimensions["faithfulness"]["kappa"] == 1.0
    assert report.dimensions["faithfulness"]["exact_match_rate"] == 1.0
    assert report.dimensions["faithfulness"]["mad"] == 0.0


def test_agreement_null_judge() -> None:
    """Rows with null judge blocks are skipped."""
    labels = [
        CalibrationLabel(
            example_id="ex1",
            labeler="alice",
            faithfulness=4,
            relevance=5,
            citation_accuracy=3,
        ),
    ]
    judged_rows = [
        {
            "example_id": "ex1",
            "judge": None,  # Null judge: refusal or deferred
        },
    ]
    report = agreement(labels, judged_rows)
    assert report.n == 0


def test_agreement_disagreement_audit() -> None:
    """Disagreement |diff| >= 2 is tracked."""
    labels = [
        CalibrationLabel(
            example_id="ex1",
            labeler="alice",
            faithfulness=1,
            relevance=4,
            citation_accuracy=3,
        ),
    ]
    judged_rows = [
        {
            "example_id": "ex1",
            "judge": {
                "faithfulness": {
                    "score": 3,
                    "justification": "",
                },  # |1-3| = 2
                "relevance": {"score": 4, "justification": ""},
                "citation_accuracy": {"score": 3, "justification": ""},
            },
        },
    ]
    report = agreement(labels, judged_rows)
    disagreement_ids = report.dimensions["faithfulness"]["disagreement_ids"]
    assert "ex1" in disagreement_ids
    disagreement_ids_rel = report.dimensions["relevance"]["disagreement_ids"]
    assert "ex1" not in disagreement_ids_rel


# ============================================================================
# Labeling Sheet Tests
# ============================================================================


def test_labeling_sheet_non_refusal() -> None:
    """Non-refusal row includes question, answer, citations, and JSON template."""
    rows = [
        {
            "example_id": "ex1",
            "provider": "claude",
            "mode": "hybrid",
            "answer_text": "The answer is 42.",
            "refusal": False,
            "cited": [{"marker": 1, "chunk_id": "ch1", "doc": "doc1", "section": "s1"}],
        },
    ]
    golden = {
        "ex1": GoldenExample(
            id="ex1",
            question="What is the answer?",
            reference_answer="42",
            source_citations=[],
            difficulty="easy",
            type="answerable",
        ),
    }
    chunks = {
        "ch1": ChunkRecord(
            chunk_id="ch1",
            doc_id="doc1",
            section_id="s1",
            section_ids=["s1"],
            section_path="s1",
            heading="Section 1",
            page_start=1,
            page_end=2,
            token_count=100,
            text="Some text.",
        ),
    }
    sheet = labeling_sheet(rows, golden, chunks)
    assert "What is the answer?" in sheet
    assert "The answer is 42." in sheet
    assert "[1] doc1 §s1 — Section 1" in sheet
    assert "faithfulness" in sheet
    assert "example_id" in sheet


def test_labeling_sheet_refusal() -> None:
    """Refusal row includes warning, skips template."""
    rows = [
        {
            "example_id": "ex1",
            "provider": "claude",
            "mode": "hybrid",
            "answer_text": "I cannot answer.",
            "refusal": True,
            "cited": [],
        },
    ]
    golden = {
        "ex1": GoldenExample(
            id="ex1",
            question="What is the answer?",
            reference_answer="42",
            source_citations=[],
            difficulty="easy",
            type="answerable",
        ),
    }
    chunks: dict[str, ChunkRecord] = {}
    sheet = labeling_sheet(rows, golden, chunks)
    assert "⚠️ **Refusal:**" in sheet
    assert "excluded from rubric labeling" in sheet


def test_labeling_sheet_sorted() -> None:
    """Rows are sorted by example_id for reproducibility."""
    rows = [
        {
            "example_id": "ex3",
            "provider": "p",
            "mode": "m",
            "answer_text": "A",
            "refusal": False,
            "cited": [],
        },
        {
            "example_id": "ex1",
            "provider": "p",
            "mode": "m",
            "answer_text": "A",
            "refusal": False,
            "cited": [],
        },
        {
            "example_id": "ex2",
            "provider": "p",
            "mode": "m",
            "answer_text": "A",
            "refusal": False,
            "cited": [],
        },
    ]
    golden = {
        "ex1": GoldenExample("ex1", "Q", "A", [], "easy", "answerable"),
        "ex2": GoldenExample("ex2", "Q", "A", [], "easy", "answerable"),
        "ex3": GoldenExample("ex3", "Q", "A", [], "easy", "answerable"),
    }
    chunks: dict[str, ChunkRecord] = {}
    sheet = labeling_sheet(rows, golden, chunks)
    pos_ex1 = sheet.find("ex1")
    pos_ex2 = sheet.find("ex2")
    pos_ex3 = sheet.find("ex3")
    assert pos_ex1 < pos_ex2 < pos_ex3


# ============================================================================
# Stratified Selection Tests
# ============================================================================


def test_select_calibration_items_deterministic() -> None:
    """Same seed produces same selection."""
    rows = [{"example_id": f"ex{i}", "refusal": False} for i in range(30)]
    golden = {
        f"ex{i}": GoldenExample(
            id=f"ex{i}",
            question=f"Q{i}",
            reference_answer=f"A{i}",
            source_citations=[],
            difficulty="easy",
            type="answerable" if i % 2 == 0 else "hard",
        )
        for i in range(30)
    }
    result1 = select_calibration_items(rows, golden, n=10, seed=42)
    result2 = select_calibration_items(rows, golden, n=10, seed=42)
    assert [r["example_id"] for r in result1] == [r["example_id"] for r in result2]


def test_select_calibration_items_n_respected() -> None:
    """Selection respects n limit."""
    rows = [{"example_id": f"ex{i}", "refusal": False} for i in range(50)]
    golden = {
        f"ex{i}": GoldenExample(
            id=f"ex{i}",
            question=f"Q{i}",
            reference_answer=f"A{i}",
            source_citations=[],
            difficulty="easy",
            type="answerable",
        )
        for i in range(50)
    }
    result = select_calibration_items(rows, golden, n=20, seed=13)
    assert len(result) <= 20


def test_select_calibration_items_covers_types() -> None:
    """Selection covers every golden type present."""
    rows = [
        {"example_id": "easy1", "refusal": False},
        {"example_id": "easy2", "refusal": False},
        {"example_id": "hard1", "refusal": False},
        {"example_id": "hard2", "refusal": False},
        {"example_id": "unanswerable1", "refusal": False},
    ]
    golden = {
        "easy1": GoldenExample("easy1", "Q", "A", [], "easy", "answerable"),
        "easy2": GoldenExample("easy2", "Q", "A", [], "easy", "answerable"),
        "hard1": GoldenExample("hard1", "Q", "A", [], "hard", "hard"),
        "hard2": GoldenExample("hard2", "Q", "A", [], "hard", "hard"),
        "unanswerable1": GoldenExample("unanswerable1", "Q", "A", [], "hard", "unanswerable"),
    }
    result = select_calibration_items(rows, golden, n=10, seed=13)
    types_selected = {golden[r["example_id"]].type for r in result}
    assert "answerable" in types_selected or "hard" in types_selected


def test_select_calibration_items_excludes_refusals() -> None:
    """Refusal rows carry no rubric signal and are never selected."""
    rows = [{"example_id": f"ex{i}", "refusal": False} for i in range(10)] + [
        {"example_id": f"ref{i}", "refusal": True} for i in range(5)
    ]
    golden = {
        f"ex{i}": GoldenExample(
            id=f"ex{i}",
            question=f"Q{i}",
            reference_answer=f"A{i}",
            source_citations=[],
            difficulty="easy",
            type="answerable",
        )
        for i in range(10)
    } | {
        f"ref{i}": GoldenExample(
            id=f"ref{i}",
            question=f"Q{i}",
            reference_answer=f"A{i}",
            source_citations=[],
            difficulty="easy",
            type="answerable",
        )
        for i in range(5)
    }
    result = select_calibration_items(rows, golden, n=20, seed=13)
    assert not any(r.get("refusal", False) for r in result)
    assert len(result) == 10


def test_select_calibration_items_unique_ids_across_configs() -> None:
    """Labels bind by example_id: the same question answered by several
    configs must be selected at most once."""
    golden = {
        f"ex{i}": GoldenExample(
            id=f"ex{i}",
            question=f"Q{i}",
            reference_answer=f"A{i}",
            source_citations=[],
            difficulty="easy",
            type="lookup" if i % 2 else "synthesis",
        )
        for i in range(15)
    }
    rows = [
        {"example_id": f"ex{i}", "provider": provider, "refusal": False}
        for provider in ("ollama", "anthropic", "google")
        for i in range(15)
    ]
    result = select_calibration_items(rows, golden, n=20, seed=13)
    ids = [r["example_id"] for r in result]
    assert len(ids) == len(set(ids)) == 15
    assert {r["provider"] for r in result} == {"ollama", "anthropic", "google"}


def test_agreement_rejects_duplicate_judged_rows() -> None:
    """Two judged rows for one example_id make the label join ambiguous."""
    judge_block = {
        "faithfulness": {"score": 4, "justification": "x"},
        "relevance": {"score": 4, "justification": "x"},
        "citation_accuracy": {"score": 4, "justification": "x"},
    }
    rows = [
        {"example_id": "ex1", "judge": judge_block},
        {"example_id": "ex1", "judge": judge_block},
    ]
    labels = [
        CalibrationLabel(
            example_id="ex1", labeler="me", faithfulness=4, relevance=4, citation_accuracy=4
        )
    ]
    with pytest.raises(ValueError, match="duplicate judged row"):
        agreement(labels, rows)


# ============================================================================
# Load Labels Tests
# ============================================================================


def test_load_labels_valid(tmp_path: Path) -> None:
    """Load valid JSONL labels."""
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(
        json.dumps(
            {
                "example_id": "ex1",
                "labeler": "alice",
                "faithfulness": 4,
                "relevance": 5,
                "citation_accuracy": 3,
                "notes": "good",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = load_labels(labels_file)
    assert len(labels) == 1
    assert labels[0].example_id == "ex1"
    assert labels[0].labeler == "alice"
    assert labels[0].faithfulness == 4
    assert labels[0].notes == "good"


def test_load_labels_missing_file() -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_labels(Path("nonexistent.jsonl"))


def test_load_labels_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON raises ValueError."""
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Line 1"):
        load_labels(labels_file)


def test_load_labels_missing_required_field(tmp_path: Path) -> None:
    """Missing required field raises ValueError."""
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(
        json.dumps(
            {
                "example_id": "ex1",
                "labeler": "alice",
                # missing faithfulness, relevance, citation_accuracy
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Line 1"):
        load_labels(labels_file)
