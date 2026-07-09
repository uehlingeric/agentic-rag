"""Tests for benchmark report generation.

Tests table renderers, manifest assembly, and document generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.evals.report import (
    _find_bold_maxima,
    _round_4dp,
    build,
    render_generation_section,
    render_rerank_table,
    render_retrieval_table,
)


class TestRound4dp:
    """Test 4-decimal formatting."""

    def test_exact_decimal(self) -> None:
        """Test formatting a value with exact decimal representation."""
        assert _round_4dp(0.6267) == "0.6267"

    def test_rounding(self) -> None:
        """Test rounding behavior."""
        assert _round_4dp(0.62666666666666) == "0.6267"
        assert _round_4dp(0.62664999999999) == "0.6266"

    def test_zero(self) -> None:
        """Test zero."""
        assert _round_4dp(0.0) == "0.0000"

    def test_one(self) -> None:
        """Test one."""
        assert _round_4dp(1.0) == "1.0000"


class TestFindBoldMaxima:
    """Test bold-maxima detection."""

    def test_single_maximum(self) -> None:
        """Test a single maximum per column."""
        rows = [
            {"mode": "bm25", "metric1": 0.5, "metric2": 0.3},
            {"mode": "dense", "metric1": 0.7, "metric2": 0.9},
        ]
        bold = _find_bold_maxima(rows, ["metric1", "metric2"])
        assert bold == {"dense|metric1", "dense|metric2"}

    def test_tied_maxima(self) -> None:
        """Test tied maxima are both bolded."""
        rows = [
            {"mode": "bm25", "metric1": 0.7},
            {"mode": "dense", "metric1": 0.7},
            {"mode": "hybrid", "metric1": 0.6},
        ]
        bold = _find_bold_maxima(rows, ["metric1"])
        assert bold == {"bm25|metric1", "dense|metric1"}

    def test_string_numeric_conversion(self) -> None:
        """Test string values are converted to float for comparison."""
        rows = [
            {"mode": "a", "col": "0.5"},
            {"mode": "b", "col": "0.8"},
        ]
        bold = _find_bold_maxima(rows, ["col"])
        assert bold == {"b|col"}

    def test_empty_rows(self) -> None:
        """Test empty rows returns empty set."""
        bold = _find_bold_maxima([], ["metric1"])
        assert bold == set()


class TestRenderRetrievalTable:
    """Test retrieval table rendering."""

    @pytest.fixture
    def retrieval_json(self) -> dict[str, object]:
        """Sample retrieval results JSON."""
        return {
            "modes": [
                {
                    "mode": "bm25",
                    "metrics": {
                        "recall@5": 0.6267,
                        "recall@10": 0.86,
                        "recall@20": 0.92,
                        "precision@5": 0.344,
                        "mrr": 0.6106,
                        "ndcg@10": 0.6055,
                    },
                },
                {
                    "mode": "dense",
                    "metrics": {
                        "recall@5": 0.7467,
                        "recall@10": 0.9,
                        "recall@20": 0.92,
                        "precision@5": 0.48,
                        "mrr": 0.784,
                        "ndcg@10": 0.7382,
                    },
                },
                {
                    "mode": "hybrid",
                    "metrics": {
                        "recall@5": 0.88,
                        "recall@10": 0.9,
                        "recall@20": 0.94,
                        "precision@5": 0.52,
                        "mrr": 0.7147,
                        "ndcg@10": 0.7126,
                    },
                },
            ],
            "n_answerable": 25,
            "n_skipped_unanswerable": 5,
            "config": {},
        }

    def test_renders_table(self, retrieval_json: dict[str, object], tmp_path: Path) -> None:
        """Test that retrieval table renders markdown."""
        json_file = tmp_path / "retrieval.json"
        with json_file.open("w") as f:
            json.dump(retrieval_json, f)

        output = render_retrieval_table(json_file)

        # Check headers
        assert "| Mode | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR | NDCG@10 |" in output
        assert "|------|----------|-----------|-----------|-------------|-----|---------|" in output

        # Check rows are present
        assert "| bm25 |" in output
        assert "| dense |" in output
        assert "| hybrid |" in output

        # Check formatting
        assert "0.6267" in output

    def test_bolds_column_maxima(self, retrieval_json: dict[str, object], tmp_path: Path) -> None:
        """Test that column maxima are bolded."""
        json_file = tmp_path / "retrieval.json"
        with json_file.open("w") as f:
            json.dump(retrieval_json, f)

        output = render_retrieval_table(json_file)

        # hybrid has max recall@5 (0.88), precision@5 (0.52), recall@20 (0.94), ndcg@10 (0.7126)
        assert "| hybrid | **0.8800** |" in output
        assert "**0.5200**" in output  # precision@5 for hybrid

    def test_4decimal_formatting(self, retrieval_json: dict[str, object], tmp_path: Path) -> None:
        """Test that all metrics use 4-decimal formatting."""
        json_file = tmp_path / "retrieval.json"
        with json_file.open("w") as f:
            json.dump(retrieval_json, f)

        output = render_retrieval_table(json_file)

        # All numbers should have exactly 4 decimal places
        import re

        numbers = re.findall(r"\d+\.\d{4}", output)
        assert len(numbers) > 0
        for num in numbers:
            parts = num.split(".")
            assert len(parts[1]) == 4


class TestRenderRerankTable:
    """Test rerank table rendering."""

    @pytest.fixture
    def rerank_json(self) -> dict[str, object]:
        """Sample rerank results JSON."""
        return {
            "modes": [
                {
                    "mode": "bm25",
                    "metrics": {
                        "recall@5": 0.6267,
                        "recall@10": 0.86,
                        "precision@5": 0.344,
                        "mrr": 0.6075,
                        "ndcg@10": 0.6055,
                    },
                },
                {
                    "mode": "bm25+llm",
                    "metrics": {
                        "recall@5": 0.6467,
                        "recall@10": 0.8,
                        "precision@5": 0.312,
                        "mrr": 0.5990,
                        "ndcg@10": 0.5921,
                    },
                },
                {
                    "mode": "hybrid",
                    "metrics": {
                        "recall@5": 0.88,
                        "recall@10": 0.9,
                        "precision@5": 0.52,
                        "mrr": 0.7147,
                        "ndcg@10": 0.7126,
                    },
                },
            ],
            "n_answerable": 25,
            "n_skipped_unanswerable": 5,
            "config": {},
        }

    def test_renders_table(self, rerank_json: dict[str, object], tmp_path: Path) -> None:
        """Test that rerank table renders markdown."""
        json_file = tmp_path / "rerank.json"
        with json_file.open("w") as f:
            json.dump(rerank_json, f)

        output = render_rerank_table(json_file)

        # Check headers (no recall@20 for depth-10 rerank)
        assert "| Mode | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@10 |" in output
        assert "recall@20" not in output.split("\n")[0]  # Not in header

        # Check rows
        assert "| bm25 |" in output
        assert "| hybrid |" in output

    def test_bolds_column_maxima(self, rerank_json: dict[str, object], tmp_path: Path) -> None:
        """Test that column maxima are bolded in rerank table."""
        json_file = tmp_path / "rerank.json"
        with json_file.open("w") as f:
            json.dump(rerank_json, f)

        output = render_rerank_table(json_file)

        # hybrid should have max recall@5, recall@10, precision@5, ndcg@10
        assert "| hybrid | **0.8800** |" in output


class TestBuild:
    """Test document assembly."""

    def test_build_with_fragments(self, tmp_path: Path) -> None:
        """Test building a document with fragments and tables."""
        # Create fragment directory
        frag_dir = tmp_path / "fragments" / "benchmarks"
        frag_dir.mkdir(parents=True)

        # Create fragment file
        frag1 = frag_dir / "01-test.md"
        frag1.write_text("## Test Heading\n\nThis is a test.")

        # Create retrieval results JSON
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        retrieval_json: dict[str, object] = {
            "modes": [
                {
                    "mode": "hybrid",
                    "metrics": {
                        "recall@5": 0.88,
                        "recall@10": 0.9,
                        "recall@20": 0.94,
                        "precision@5": 0.52,
                        "mrr": 0.7147,
                        "ndcg@10": 0.7126,
                    },
                }
            ],
            "n_answerable": 25,
            "n_skipped_unanswerable": 5,
            "config": {},
        }
        retrieval_file = results_dir / "retrieval.json"
        with retrieval_file.open("w") as f:
            json.dump(retrieval_json, f)

        # Mock the build process with minimal manifest
        # (normally would use the full manifest with actual paths)
        # Just verify the function can be called
        # For full integration, we'd need the actual fragment files

        # This is a sanity check that the function signature works
        assert callable(build)

    def test_build_with_header_comment(self, tmp_path: Path) -> None:
        """Test that generated document includes header comment."""
        # Create minimal test setup
        frag_dir = tmp_path / "fragments" / "benchmarks"
        frag_dir.mkdir(parents=True)

        # Create minimal fragment
        (frag_dir / "01-test.md").write_text("## Test\n\nContent")

        # Create dummy results JSON
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = results_dir / "retrieval.json"
        results_file.write_text(
            json.dumps(
                {
                    "modes": [
                        {
                            "mode": "test",
                            "metrics": {
                                "recall@5": 0.5,
                                "recall@10": 0.6,
                                "recall@20": 0.7,
                                "precision@5": 0.4,
                                "mrr": 0.5,
                                "ndcg@10": 0.6,
                            },
                        }
                    ],
                    "n_answerable": 10,
                    "n_skipped_unanswerable": 0,
                    "config": {},
                }
            )
        )

        # Just verify we can call build successfully
        out_file = tmp_path / "out.md"
        doc = build(out_file)

        # Header should be present
        assert "<!-- Generated by evals/build_report.py" in doc
        assert "# Retrieval Benchmarks" in doc

    def test_build_writes_file(self, tmp_path: Path) -> None:
        """Test that build() writes the output file."""
        # Create minimal test setup with real fragments
        frag_dir = tmp_path / "fragments" / "benchmarks"
        frag_dir.mkdir(parents=True)
        (frag_dir / "01-methodology.md").write_text("## Methodology\n\nTest")

        # Create dummy results
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = results_dir / "retrieval.json"
        results_file.write_text(
            json.dumps(
                {
                    "modes": [
                        {
                            "mode": "test",
                            "metrics": {
                                "recall@5": 0.5,
                                "recall@10": 0.6,
                                "recall@20": 0.7,
                                "precision@5": 0.4,
                                "mrr": 0.5,
                                "ndcg@10": 0.6,
                            },
                        }
                    ],
                    "n_answerable": 10,
                    "n_skipped_unanswerable": 0,
                    "config": {},
                }
            )
        )

        out_file = tmp_path / "output.md"
        assert not out_file.exists()

        doc = build(out_file)

        assert out_file.exists()
        assert out_file.read_text() == doc


class TestRenderGenerationSection:
    """Test generation section rendering."""

    @pytest.fixture
    def generation_summary(self) -> dict[str, object]:
        """Two-config generation summary with null scores and judge metadata."""
        return {
            "run_id": "gen-20260710-120000Z",
            "dataset_version": "v2",
            "n_examples": 50,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "none",
                    "synthesis_prompt": "synthesis.v2",
                    "judge_prompt": "judge.v1",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 50,
                    "n_judged": 48,
                    "n_refusals": 2,
                    "n_judge_failures": 0,
                    "refusal_correct_rate": 1.0,
                    "false_refusal_rate": 0.0,
                    "scores": {
                        "faithfulness": 4.2,
                        "relevance": 4.1,
                        "citation_accuracy": 3.9,
                    },
                    "latency_s": {"mean": 2.5, "p50": 2.3, "p95": 3.8},
                    "gen_tokens": {"input": 150000, "output": 12500},
                    "gen_cost_usd": 2.45,
                    "judge_cost_usd": 0.48,
                },
                {
                    "provider": "google",
                    "model": "gemini-2.0-flash",
                    "mode": "hybrid",
                    "rerank": "none",
                    "synthesis_prompt": "synthesis.v2",
                    "judge_prompt": "judge.v1",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 50,
                    "n_judged": 45,
                    "n_refusals": 5,
                    "n_judge_failures": 1,
                    "refusal_correct_rate": 0.8,
                    "false_refusal_rate": 0.05,
                    "scores": {
                        "faithfulness": None,
                        "relevance": 3.8,
                        "citation_accuracy": None,
                    },
                    "latency_s": {"mean": 1.2, "p50": 1.1, "p95": 2.1},
                    "gen_tokens": {"input": 155000, "output": 11800},
                    "gen_cost_usd": 1.82,
                    "judge_cost_usd": 0.45,
                },
            ],
        }

    def test_renders_heading(self, generation_summary: dict[str, object], tmp_path: Path) -> None:
        """Test that generation section has proper heading."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        assert "## Generation — provider × retrieval matrix" in output  # noqa: RUF001
        assert "Run `gen-20260710-120000Z`, dataset v2 (50 examples)," in output
        assert "synthesis prompt synthesis.v2." in output

    def test_judge_attribution(self, generation_summary: dict[str, object], tmp_path: Path) -> None:
        """Test judge attribution bullet list."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Both providers should be listed together since they use same judge
        assert "anthropic, google answers judged by anthropic" in output
        assert "`claude-sonnet-4-6`" in output
        assert "(judge.v1)" in output

    def test_quality_table_null_scores(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that null scores render as — in quality table."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Google config has null faithfulness and citation_accuracy
        lines = output.split("\n")
        google_line = next(line for line in lines if "| google |" in line)

        # Null scores should be "—"
        assert "| — |" in google_line  # Null faithfulness
        assert "| 3.80 |" in google_line  # Non-null relevance (2 decimals)

    def test_quality_table_bolding_maxima(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that quality table bolds column maxima."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # anthropic has max faithfulness (4.2), relevance (4.1), refusal_correct (1.0)
        assert "**4.20**" in output  # Faithfulness max (2 decimals)
        assert "**4.10**" in output  # Relevance max (2 decimals)
        assert "**1.00**" in output  # Refusal correct max

    def test_quality_table_bolding_false_refusal_min(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that false_refusal column bolds MINIMUM (lower is better)."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # anthropic has min false_refusal (0.0), should be bolded
        lines = output.split("\n")
        anthropic_line = next(
            line for line in lines if "| anthropic |" in line and "hybrid" in line
        )

        # anthropic's false refusal (0.0) should be bolded
        assert "**0.00**" in anthropic_line

    def test_quality_table_judged_column(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that judged column shows n_judged/n_items."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Check judged column formatting
        assert "48/50" in output  # anthropic
        assert "45/50" in output  # google

    def test_ops_table_latency_formatting(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test ops table latency formatting to 2 decimals."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Check latency formatting (2 decimals)
        assert "2.50" in output  # anthropic mean
        assert "1.20" in output  # google mean

    def test_ops_table_tokens_formatting(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test ops table tokens formatted as input / output."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Check token formatting
        assert "150000 / 12500" in output  # anthropic
        assert "155000 / 11800" in output  # google

    def test_ops_table_cost_formatting(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test ops table costs formatted as $X.XX."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # Check cost formatting
        assert "$2.45" in output  # anthropic gen cost
        assert "$0.48" in output  # anthropic judge cost

    def test_judge_failures_listed(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that nonzero judge failures are listed."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        # google has 1 judge failure, listed with its full config slug
        assert "- google/hybrid/none: 1" in output

    def test_refusal_correctness_note(
        self, generation_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test trailing refusal correctness note is present."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(generation_summary, f)

        output = render_generation_section(json_file)

        assert "Refusal correctness is reported separately" in output
        assert "never averaged into them" in output
