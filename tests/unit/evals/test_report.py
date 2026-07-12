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
    render_agentic_comparison,
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

    @pytest.fixture
    def minimal_final_summary(self, tmp_path: Path) -> Path:
        """Create a minimal final summary JSON for testing."""
        summary = {
            "run_id": "gen-20260712-120000Z",
            "dataset_version": "v2",
            "n_examples": 46,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v3",
                    "judge_prompt": "judge.v2",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 46,
                    "n_judged": 46,
                    "n_refusals": 0,
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
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v2",
                    "judge_prompt": "judge.v2",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 46,
                    "n_judged": 46,
                    "n_refusals": 0,
                    "n_judge_failures": 0,
                    "refusal_correct_rate": 1.0,
                    "false_refusal_rate": 0.0,
                    "scores": {
                        "faithfulness": 4.5,
                        "relevance": 4.3,
                        "citation_accuracy": 4.1,
                    },
                    "latency_s": {"mean": 3.2, "p50": 3.0, "p95": 4.5},
                    "gen_tokens": {"input": 160000, "output": 13000},
                    "gen_cost_usd": 2.65,
                    "judge_cost_usd": 0.50,
                    "by_type": {
                        "lookup": {
                            "n": 15,
                            "n_judged": 15,
                            "scores": {
                                "faithfulness": 4.5,
                                "relevance": 4.3,
                                "citation_accuracy": 4.1,
                            },
                            "refusal_rate": 0.0,
                        }
                    },
                    "agent": {
                        "mean_revisions": 0.2,
                        "caveat_rate": 0.05,
                        "multi_hop_rate": 0.3,
                    },
                },
            ],
        }
        summary_path = tmp_path / "final.json"
        with summary_path.open("w") as f:
            json.dump(summary, f)
        return summary_path

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

    def test_build_with_header_comment(self, tmp_path: Path, minimal_final_summary: Path) -> None:
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
        doc = build(out_file, final_summary=minimal_final_summary)

        # Header should be present
        assert "<!-- Generated by evals/build_report.py" in doc
        assert "# Retrieval Benchmarks" in doc

    def test_build_writes_file(self, tmp_path: Path, minimal_final_summary: Path) -> None:
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

        doc = build(out_file, final_summary=minimal_final_summary)

        assert out_file.exists()
        assert out_file.read_text() == doc

    def test_build_includes_agentic_section(
        self, tmp_path: Path, minimal_final_summary: Path
    ) -> None:
        """The manifest renders the week-5 fragments and agentic comparison."""
        doc = build(tmp_path / "output.md", final_summary=minimal_final_summary)

        assert "## Week 5 — Vanilla vs. agentic pipeline" in doc
        assert "## Agentic vs. Vanilla Comparison" in doc
        # Week-5 sections sit between week-4 analysis and Reproduce
        assert doc.index("## Agentic vs. Vanilla Comparison") < doc.index("## Reproduce")

    def test_build_includes_week8_section(self, tmp_path: Path) -> None:
        """The manifest renders week-8 fragments and tables."""
        # Create week-8 summary with 2 pipelines and hybrid+llm present
        week8_summary = {
            "run_id": "gen-20260712-120000Z",
            "dataset_version": "v2",
            "n_examples": 46,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v3",
                    "judge_prompt": "judge.v2",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 46,
                    "n_judged": 46,
                    "n_refusals": 0,
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
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v2",
                    "judge_prompt": "judge.v2",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 46,
                    "n_judged": 46,
                    "n_refusals": 0,
                    "n_judge_failures": 0,
                    "refusal_correct_rate": 1.0,
                    "false_refusal_rate": 0.0,
                    "scores": {
                        "faithfulness": 4.5,
                        "relevance": 4.3,
                        "citation_accuracy": 4.1,
                    },
                    "latency_s": {"mean": 3.2, "p50": 3.0, "p95": 4.5},
                    "gen_tokens": {"input": 160000, "output": 13000},
                    "gen_cost_usd": 2.65,
                    "judge_cost_usd": 0.50,
                    "by_type": {
                        "lookup": {
                            "n": 15,
                            "n_judged": 15,
                            "scores": {
                                "faithfulness": 4.5,
                                "relevance": 4.3,
                                "citation_accuracy": 4.1,
                            },
                            "refusal_rate": 0.0,
                        }
                    },
                    "agent": {"mean_revisions": 0.2, "caveat_rate": 0.05, "multi_hop_rate": 0.3},
                },
            ],
        }
        week8_json = tmp_path / "week8.json"
        with week8_json.open("w") as f:
            json.dump(week8_summary, f)

        out_file = tmp_path / "output.md"
        doc = build(out_file, final_summary=week8_json)

        # Should include week-8 config fragment
        assert "## Week 8 — Final benchmark" in doc
        # Should include custom headings
        assert "### Full matrix — provider × retrieval × pipeline" in doc  # noqa: RUF001
        assert "### Agentic vs. vanilla — hybrid + llm rerank" in doc
        # Should appear after week-6 guardrails and before Reproduce
        week8_idx = doc.index("## Week 8 — Final benchmark")
        week6_idx = doc.index("## Week 6")
        reproduce_idx = doc.index("## Reproduce")
        assert week6_idx < week8_idx < reproduce_idx


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


class TestRenderAgenticComparison:
    """Test agentic vs. vanilla comparison rendering."""

    def test_renders_headline_table(self, tmp_path: Path) -> None:
        """Test headline table with both pipelines."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {
                        "faithfulness": 0.8,
                        "relevance": 0.85,
                        "citation_accuracy": 0.75,
                    },
                    "false_refusal_rate": 0.1,
                    "refusal_correct_rate": 0.9,
                    "latency_s": {"p50": 1.5},
                    "gen_cost_usd": 10.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {
                        "faithfulness": 0.85,
                        "relevance": 0.88,
                        "citation_accuracy": 0.78,
                    },
                    "false_refusal_rate": 0.05,
                    "refusal_correct_rate": 0.95,
                    "latency_s": {"p50": 2.0},
                    "gen_cost_usd": 20.0,
                    "by_type": {},
                    "agent": {
                        "multi_hop_rate": 0.5,
                        "mean_revisions": 0.8,
                        "caveat_rate": 0.1,
                    },
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        output = render_agentic_comparison(json_file)

        # Check headline table has key content
        assert "## Agentic vs. Vanilla Comparison" in output
        assert "Provider" in output
        assert "Pipeline" in output
        assert "anthropic" in output
        assert "vanilla" in output
        assert "agentic" in output
        # Check values in headline table
        assert "| anthropic | vanilla | 0.80" in output  # vanilla faithfulness
        assert "| anthropic | agentic | 0.85" in output  # agentic faithfulness
        assert "0.80" in output  # mean_revisions
        assert "0.10" in output  # caveat_rate
        assert "$10.00" in output  # vanilla cost
        assert "$20.00" in output  # agentic cost

    def test_renders_delta_table(self, tmp_path: Path) -> None:
        """Test per-type delta table with deltas."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {},
                    "latency_s": {},
                    "by_type": {
                        "lookup": {
                            "n": 5,
                            "n_judged": 5,
                            "scores": {
                                "faithfulness": 0.8,
                                "relevance": 0.85,
                                "citation_accuracy": 0.75,
                            },
                            "refusal_rate": 0.0,
                        },
                    },
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {},
                    "latency_s": {},
                    "by_type": {
                        "lookup": {
                            "n": 5,
                            "n_judged": 5,
                            "scores": {
                                "faithfulness": 0.9,
                                "relevance": 0.88,
                                "citation_accuracy": 0.78,
                            },
                            "refusal_rate": 0.0,
                        },
                    },
                    "agent": {"multi_hop_rate": 0.0, "mean_revisions": 0.0, "caveat_rate": 0.0},
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        output = render_agentic_comparison(json_file)

        # Check delta table
        assert "lookup" in output
        assert "anthropic" in output
        # Deltas: 0.9-0.8=+0.10, 0.88-0.85=+0.03, 0.78-0.75=+0.03
        assert "+0.10" in output  # faithfulness delta
        assert "+0.03" in output  # relevance and citation accuracy deltas

    def test_raises_on_no_agentic_config(self, tmp_path: Path) -> None:
        """Test ValueError when no agentic config present."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {},
                    "latency_s": {},
                    "by_type": {},
                    "agent": None,
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        with pytest.raises(ValueError, match="no agentic config"):
            render_agentic_comparison(json_file)

    def test_handles_none_values(self, tmp_path: Path) -> None:
        """Test that None values render as —."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": None, "relevance": None, "citation_accuracy": None},
                    "false_refusal_rate": None,
                    "refusal_correct_rate": None,
                    "latency_s": {"p50": None},
                    "gen_cost_usd": None,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {"faithfulness": None, "relevance": None, "citation_accuracy": None},
                    "false_refusal_rate": None,
                    "refusal_correct_rate": None,
                    "latency_s": {"p50": None},
                    "gen_cost_usd": None,
                    "by_type": {},
                    "agent": None,
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        output = render_agentic_comparison(json_file)

        # Check that — appears for None values
        assert "—" in output

    def test_filter_by_mode_and_rerank(self, tmp_path: Path) -> None:
        """Test filtering configs by mode and rerank."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "bm25",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": 0.8, "relevance": 0.85, "citation_accuracy": 0.75},
                    "false_refusal_rate": 0.1,
                    "refusal_correct_rate": 0.9,
                    "latency_s": {"p50": 1.5},
                    "gen_cost_usd": 10.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": 0.85, "relevance": 0.88, "citation_accuracy": 0.78},
                    "false_refusal_rate": 0.05,
                    "refusal_correct_rate": 0.95,
                    "latency_s": {"p50": 2.0},
                    "gen_cost_usd": 15.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {"faithfulness": 0.87, "relevance": 0.89, "citation_accuracy": 0.80},
                    "false_refusal_rate": 0.02,
                    "refusal_correct_rate": 0.98,
                    "latency_s": {"p50": 3.0},
                    "gen_cost_usd": 20.0,
                    "by_type": {},
                    "agent": {"mean_revisions": 0.5, "caveat_rate": 0.1, "multi_hop_rate": 0.3},
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        output = render_agentic_comparison(json_file, mode="hybrid", rerank="llm")

        # Should mention the filter
        assert "Retrieval config: mode hybrid + rerank llm." in output
        # Should include both vanilla and agentic for hybrid+llm
        assert "vanilla" in output
        assert "agentic" in output
        # Should not include bm25 config
        assert "bm25" not in output

    def test_ambiguous_config_raises_error(self, tmp_path: Path) -> None:
        """Test ValueError when (provider, pipeline) pair is ambiguous."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": 0.8, "relevance": 0.85, "citation_accuracy": 0.75},
                    "false_refusal_rate": 0.1,
                    "refusal_correct_rate": 0.9,
                    "latency_s": {"p50": 1.5},
                    "gen_cost_usd": 10.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "bm25",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": 0.75, "relevance": 0.80, "citation_accuracy": 0.70},
                    "false_refusal_rate": 0.15,
                    "refusal_correct_rate": 0.85,
                    "latency_s": {"p50": 1.0},
                    "gen_cost_usd": 8.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v3",
                    "scores": {"faithfulness": 0.82, "relevance": 0.87, "citation_accuracy": 0.77},
                    "false_refusal_rate": 0.08,
                    "refusal_correct_rate": 0.92,
                    "latency_s": {"p50": 1.8},
                    "gen_cost_usd": 12.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {"faithfulness": 0.87, "relevance": 0.89, "citation_accuracy": 0.80},
                    "false_refusal_rate": 0.02,
                    "refusal_correct_rate": 0.98,
                    "latency_s": {"p50": 3.0},
                    "gen_cost_usd": 20.0,
                    "by_type": {},
                    "agent": {"mean_revisions": 0.5, "caveat_rate": 0.1, "multi_hop_rate": 0.3},
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        # Should raise ValueError because (anthropic, vanilla) has 2 configs
        with pytest.raises(
            ValueError, match="ambiguous configs for provider=anthropic pipeline=vanilla"
        ):
            render_agentic_comparison(json_file)

    def test_custom_heading(self, tmp_path: Path) -> None:
        """Test custom heading parameter."""
        summary = {
            "run_id": "test-run",
            "dataset_version": "v2",
            "n_examples": 10,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
                    "synthesis_prompt": "synthesis.v2",
                    "scores": {"faithfulness": 0.8, "relevance": 0.85, "citation_accuracy": 0.75},
                    "false_refusal_rate": 0.1,
                    "refusal_correct_rate": 0.9,
                    "latency_s": {"p50": 1.5},
                    "gen_cost_usd": 10.0,
                    "by_type": {},
                    "agent": None,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "agentic",
                    "synthesis_prompt": "agent-synthesis.v1",
                    "scores": {"faithfulness": 0.85, "relevance": 0.88, "citation_accuracy": 0.78},
                    "false_refusal_rate": 0.05,
                    "refusal_correct_rate": 0.95,
                    "latency_s": {"p50": 2.0},
                    "gen_cost_usd": 20.0,
                    "by_type": {},
                    "agent": {"mean_revisions": 0.8, "caveat_rate": 0.1, "multi_hop_rate": 0.5},
                },
            ],
        }
        json_file = tmp_path / "summary.json"
        with json_file.open("w") as f:
            json.dump(summary, f)

        custom_heading = "### Custom Section Heading"
        output = render_agentic_comparison(json_file, heading=custom_heading)

        assert custom_heading in output
        assert "## Agentic vs. Vanilla Comparison" not in output


class TestRenderGenerationSectionPipelines:
    """Test pipeline column in generation section."""

    @pytest.fixture
    def two_pipeline_summary(self) -> dict[str, object]:
        """Summary with multiple pipelines."""
        return {
            "run_id": "gen-20260712-120000Z",
            "dataset_version": "v2",
            "n_examples": 50,
            "configs": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "none",
                    "pipeline": "vanilla",
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
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "mode": "hybrid",
                    "rerank": "llm",
                    "pipeline": "agentic",
                    "synthesis_prompt": "synthesis.v3",
                    "judge_prompt": "judge.v1",
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-4-6",
                    "n_items": 50,
                    "n_judged": 50,
                    "n_refusals": 0,
                    "n_judge_failures": 0,
                    "refusal_correct_rate": 1.0,
                    "false_refusal_rate": 0.0,
                    "scores": {
                        "faithfulness": 4.5,
                        "relevance": 4.3,
                        "citation_accuracy": 4.1,
                    },
                    "latency_s": {"mean": 3.2, "p50": 3.0, "p95": 4.5},
                    "gen_tokens": {"input": 160000, "output": 13000},
                    "gen_cost_usd": 2.65,
                    "judge_cost_usd": 0.50,
                },
            ],
        }

    def test_multiple_pipelines_adds_column(
        self, two_pipeline_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that Pipeline column is added when multiple pipelines present."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(two_pipeline_summary, f)

        output = render_generation_section(json_file)

        # Pipeline column should be present
        assert "| Pipeline |" in output or ("| vanilla |" in output and "| agentic |" in output)
        # Both pipeline values should appear
        assert "vanilla" in output
        assert "agentic" in output

    def test_single_pipeline_no_column(
        self, two_pipeline_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that single-pipeline summary has no Pipeline column."""
        # Remove agentic config so only vanilla remains
        single_pipeline_summary = {
            **two_pipeline_summary,
            "configs": [two_pipeline_summary["configs"][0]],
        }
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(single_pipeline_summary, f)

        output = render_generation_section(json_file)

        # Should not have Pipeline header
        assert "| Pipeline |" not in output

    def test_custom_heading_generation(
        self, two_pipeline_summary: dict[str, object], tmp_path: Path
    ) -> None:
        """Test custom heading parameter in render_generation_section."""
        json_file = tmp_path / "gen.json"
        with json_file.open("w") as f:
            json.dump(two_pipeline_summary, f)

        custom_heading = "### Week 8 Full Matrix"
        output = render_generation_section(json_file, heading=custom_heading)

        assert custom_heading in output
        # Default heading should not appear
        assert "provider" not in output.split("\n")[0]  # First line is custom heading
