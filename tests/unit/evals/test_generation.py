"""Unit tests for generation evaluation module."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.config import (
    AnthropicSettings,
    GoogleSettings,
    OllamaSettings,
    Settings,
)
from agentic_rag.evals.generation import (
    RunConfig,
    config_settings,
    estimate_cost,
    eval_set,
    run_config,
    summarize,
)
from agentic_rag.evals.retrieval import Citation, GoldenExample
from agentic_rag.pipeline.base import Answer, CitedChunk, StageTiming
from agentic_rag.providers.base import Completion, LLMProvider, Message, Usage
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


@pytest.fixture
def golden_examples() -> list[GoldenExample]:
    """Create test golden examples."""
    return [
        GoldenExample(
            id="q1",
            question="What is foo?",
            reference_answer="Foo is a bar.",
            source_citations=[Citation(doc="doc1", section="sec1")],
            difficulty="easy",
            type="answerable",
            held_out=False,
        ),
        GoldenExample(
            id="q2",
            question="What is impossible?",
            reference_answer="",
            source_citations=[],
            difficulty="hard",
            type="unanswerable",
            held_out=False,
        ),
        GoldenExample(
            id="q3",
            question="What is baz?",
            reference_answer="Baz is something.",
            source_citations=[Citation(doc="doc2", section="sec2")],
            difficulty="medium",
            type="answerable",
            held_out=False,
        ),
    ]


@pytest.fixture
def base_settings() -> Settings:
    """Create test settings."""
    return Settings(
        provider="ollama",
        anthropic=AnthropicSettings(model="claude-sonnet-5"),
        google=GoogleSettings(model="gemini-3.5-flash"),
        ollama=OllamaSettings(model="llama3.1:8b"),
    )


class StubPipeline:
    """Stub pipeline for testing: returns canned answers without network."""

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.ask_calls: list[tuple[str, Any]] = []

    async def ask(self, question: str, **kwargs: Any) -> Answer:
        """Record call and return canned answer."""
        self.ask_calls.append((question, kwargs))
        return self.answer


class StubJudgeLLM(LLMProvider):
    """Stub judge LLM that returns fixed scores or raises on demand."""

    name = "stub"

    def __init__(
        self,
        model_id: str = "stub-model",
        raise_parse_error: bool = False,
    ) -> None:
        self.model_id = model_id
        self.raise_parse_error = raise_parse_error
        self.complete_calls: list[Sequence[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Record call and return canned completion or raise."""
        self.complete_calls.append(messages)

        if self.raise_parse_error:
            # Return invalid JSON to trigger parse error
            return Completion(
                text="This is not valid JSON {",
                model=model or self.model_id,
                usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
            )

        # Return valid judge JSON
        judge_json = """{
            "faithfulness": {"score": 5, "justification": "Perfect"},
            "relevance": {"score": 5, "justification": "Perfect"},
            "citation_accuracy": {"score": 5, "justification": "Perfect"}
        }"""
        return Completion(
            text=judge_json,
            model=model or self.model_id,
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        )

    def count_tokens(self, text: str) -> int:
        """Stub token counter."""
        return len(text) // 4


def test_run_config_slug() -> None:
    """Test RunConfig slug generation."""
    cfg = RunConfig(provider="anthropic", mode="hybrid", rerank="llm")
    assert cfg.slug() == "anthropic--hybrid--llm--vanilla"

    cfg_agentic = RunConfig(provider="anthropic", mode="hybrid", rerank="llm", pipeline="agentic")
    assert cfg_agentic.slug() == "anthropic--hybrid--llm--agentic"


def test_config_settings(base_settings: Settings) -> None:
    """Test config_settings swaps provider and rerank mode."""
    cfg = RunConfig(provider="anthropic", mode="hybrid", rerank="cross-encoder")
    result = config_settings(base_settings, cfg)

    # Provider should be swapped
    assert result.provider == "anthropic"
    # Rerank mode should be swapped
    assert result.rerank.mode == "cross-encoder"
    # Other settings should be unchanged
    assert result.data_dir == base_settings.data_dir


def test_estimate_cost(base_settings: Settings) -> None:
    """Test cost estimation."""
    configs = [
        RunConfig(provider="ollama", mode="bm25", rerank="none"),
        RunConfig(provider="anthropic", mode="hybrid", rerank="llm"),
    ]

    results = estimate_cost(configs, n_examples=100, settings=base_settings)

    assert len(results) == 2
    cfg0, cost0 = results[0]
    cfg1, cost1 = results[1]

    # Ollama generation is free but judge (anthropic) costs 0.75
    assert cfg0.provider == "ollama"
    assert cost0 == 0.75

    # Anthropic should have a higher cost (generation + google judging)
    assert cfg1.provider == "anthropic"
    assert cost1 is not None and cost1 > cost0


def test_estimate_cost_unknown_model(base_settings: Settings) -> None:
    """Test cost estimation with unknown models returns None."""
    # Modify settings to use unknown model
    modified_settings = base_settings.model_copy(
        update={
            "anthropic": base_settings.anthropic.model_copy(update={"model": "unknown-model-xyz"})
        }
    )

    configs = [RunConfig(provider="anthropic", mode="hybrid", rerank="none")]
    results = estimate_cost(configs, n_examples=100, settings=modified_settings)

    _, cost = results[0]
    assert cost is None


def test_estimate_cost_judge_pricing_regression(base_settings: Settings) -> None:
    """Regression: estimate_cost must price judge even for ollama configs.

    Bug #1: judge_total = None; if judge_total is None: continue made judge
    branch unreachable, so no estimate ever included judge cost. Fixed to
    short-circuit on gen_total is None and always price judge.
    """
    # Ollama config with judge preference including anthropic
    cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none")
    results = estimate_cost([cfg], n_examples=100, settings=base_settings)

    _, cost = results[0]
    # Ollama generation is free ($0 per example * 100 = $0)
    # But judge is anthropic pricing: EST_JUDGE_INPUT=2500, EST_JUDGE_OUTPUT=250
    # Claude Sonnet: 2.0 input, 10.0 output per million tokens
    # (2500/1M * 2.0) + (250/1M * 10.0) = 0.005 + 0.0025 = 0.0075 per example
    # 0.0075 * 100 = 0.75
    assert cost == 0.75

    # Anthropic config should price google judge
    cfg2 = RunConfig(provider="anthropic", mode="hybrid", rerank="none")
    results2 = estimate_cost([cfg2], n_examples=100, settings=base_settings)
    _, cost2 = results2[0]
    # Anthropic generation: (6500/1M * 2.0) + (300/1M * 10.0) = 0.013 + 0.003 = 0.016
    # Google judge: 1.5 input, 9.0 output per million tokens
    # (2500/1M * 1.5) + (250/1M * 9.0) = 0.00375 + 0.00225 = 0.006 per example
    # 0.016 + 0.006 = 0.022 per example
    # 0.022 * 100 = 2.2
    assert cost2 == 2.2


async def test_run_config_resume_skips_existing(
    golden_examples: list[GoldenExample],
) -> None:
    """Test that run_config resumes and skips existing examples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"

        # Pre-write two results
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"example_id": "q1", "dataset_version": "v2"}) + "\n")
            f.write(json.dumps({"example_id": "q2", "dataset_version": "v2"}) + "\n")

        # Create canned answer
        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            section_id="sec1",
            section_ids=["sec1"],
            section_path="doc1/sec1",
            heading="Section 1",
            page_start=1,
            page_end=1,
            token_count=100,
            text="Text 1",
        )
        answer = Answer(
            text="Answer to q3",
            citations=[CitedChunk(marker=1, chunk=chunk)],
            context=[ScoredChunk(chunk=chunk, score=0.9, rank=1)],
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.01),
            timings=[StageTiming(stage="synthesis", seconds=0.5)],
            refusal=False,
            invalid_citations=[],
        )
        pipeline = StubPipeline(answer)

        cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none")
        settings = Settings()

        await run_config(
            cfg,
            golden_examples,
            settings,
            out_path,
            dataset_version="v2",
            concurrency=1,
            do_judge=False,
            _pipeline_factory=lambda s: pipeline,
        )

        # Should have 3 total lines (2 existing + 1 new)
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # The new one should be q3 and carry dataset_version
        new_row = json.loads(lines[-1])
        assert new_row["example_id"] == "q3"
        assert new_row["dataset_version"] == "v2"


async def test_run_config_refusal_no_judge() -> None:
    """Test that refusal rows get judge=None without judge call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"

        # Create one golden example
        golden = [
            GoldenExample(
                id="q1",
                question="What?",
                reference_answer="",
                source_citations=[],
                difficulty="hard",
                type="unanswerable",
            ),
        ]

        # Create refusal answer
        answer = Answer(
            text="",
            citations=[],
            context=[],
            usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0),
            timings=[StageTiming(stage="synthesis", seconds=0.1)],
            refusal=True,
            invalid_citations=[],
        )
        pipeline = StubPipeline(answer)

        cfg = RunConfig(provider="ollama", mode="bm25", rerank="none")
        settings = Settings()

        await run_config(
            cfg,
            golden,
            settings,
            out_path,
            dataset_version="v2",
            concurrency=1,
            do_judge=True,  # Judge enabled, but refusal should skip it
            _pipeline_factory=lambda s: pipeline,
        )

        # Read result
        result = json.loads(out_path.read_text().strip())
        assert result["refusal"] is True
        assert result["judge"] is None


async def test_run_config_guardrails_production_path(
    golden_examples: list[GoldenExample],
) -> None:
    """guardrails=True wraps the pipeline in GuardedPipeline.

    Rows carry guardrails=True plus guardrails_in/guardrails_out stage
    timings; a clean answer passes through unmodified. Without the flag,
    rows carry guardrails=False and no guardrail stages.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            section_id="sec1",
            section_ids=["sec1"],
            section_path="doc1/sec1",
            heading="Section 1",
            page_start=1,
            page_end=1,
            token_count=100,
            text="Text 1",
        )
        answer = Answer(
            text="Foo is a bar per the spec.",
            citations=[CitedChunk(marker=1, chunk=chunk)],
            context=[ScoredChunk(chunk=chunk, score=0.9, rank=1)],
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.01),
            timings=[StageTiming(stage="synthesis", seconds=0.5)],
            refusal=False,
            invalid_citations=[],
        )
        settings = Settings(data_dir=Path(tmpdir) / "data")
        pipeline = StubPipeline(answer)
        # GuardedPipeline reads inner.settings.rerank.mode for audit records
        pipeline.settings = settings  # type: ignore[attr-defined]

        cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none")

        guarded_path = Path(tmpdir) / "guarded.jsonl"
        await run_config(
            cfg,
            golden_examples[:1],
            settings,
            guarded_path,
            dataset_version="v2",
            concurrency=1,
            do_judge=False,
            guardrails=True,
            _pipeline_factory=lambda s: pipeline,
        )

        row = json.loads(guarded_path.read_text().strip())
        assert row["guardrails"] is True
        assert row["answer_text"] == "Foo is a bar per the spec."
        assert row["refusal"] is False
        assert "guardrails_in" in row["latency_s"]
        assert "guardrails_out" in row["latency_s"]
        # Inner pipeline received the (unredacted) question
        assert pipeline.ask_calls[0][0] == "What is foo?"

        plain_path = Path(tmpdir) / "plain.jsonl"
        await run_config(
            cfg,
            golden_examples[:1],
            settings,
            plain_path,
            dataset_version="v2",
            concurrency=1,
            do_judge=False,
            _pipeline_factory=lambda s: pipeline,
        )

        plain_row = json.loads(plain_path.read_text().strip())
        assert plain_row["guardrails"] is False
        assert "guardrails_in" not in plain_row["latency_s"]


async def test_run_config_judge_parse_error() -> None:
    """Test that judge parse error is caught and row written with judge=None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"

        # Create one golden example
        golden = [
            GoldenExample(
                id="q1",
                question="What is foo?",
                reference_answer="Foo.",
                source_citations=[Citation(doc="doc1", section="sec1")],
                difficulty="easy",
                type="answerable",
            ),
        ]

        # Create answer with citation
        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            section_id="sec1",
            section_ids=["sec1"],
            section_path="doc1/sec1",
            heading="Section 1",
            page_start=1,
            page_end=1,
            token_count=100,
            text="Text 1",
        )
        answer = Answer(
            text="Foo is...",
            citations=[CitedChunk(marker=1, chunk=chunk)],
            context=[ScoredChunk(chunk=chunk, score=0.9, rank=1)],
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.01),
            timings=[StageTiming(stage="synthesis", seconds=0.5)],
            refusal=False,
            invalid_citations=[],
        )
        pipeline = StubPipeline(answer)

        # Create judge that raises parse error
        stub_judge = StubJudgeLLM(raise_parse_error=True)

        cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none")
        settings = Settings()

        # Monkeypatch get_llm_provider to return stub judge
        import agentic_rag.evals.generation as gen_module

        orig_get_llm = gen_module.get_llm_provider

        def mock_get_llm(provider: str, s: Settings) -> object:
            if provider == "anthropic":
                return stub_judge
            return orig_get_llm(provider, s)

        gen_module.get_llm_provider = mock_get_llm
        try:
            await run_config(
                cfg,
                golden,
                settings,
                out_path,
                dataset_version="v2",
                concurrency=1,
                do_judge=True,
                _pipeline_factory=lambda s: pipeline,
            )
        finally:
            gen_module.get_llm_provider = orig_get_llm

        # Read result - judge should be None due to parse error
        result = json.loads(out_path.read_text().strip())
        assert result["refusal"] is False
        assert result["judge"] is None


def test_summarize_basic(golden_examples: list[GoldenExample]) -> None:
    """Test summarize aggregation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create test rows
        rows: list[dict[str, object]] = [
            {
                "example_id": "q1",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "synthesis_prompt": "synthesis.v2",
                "dataset_version": "v2",
                "answer_text": "Answer to q1",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 3,
                "gen_usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.0,
                },
                "latency_s": {"synthesis": 0.5, "total": 0.5},
                "judge": {
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-5",
                    "prompt_id": "judge.v1",
                    "faithfulness": {"score": 5, "justification": "Perfect"},
                    "relevance": {"score": 5, "justification": "Perfect"},
                    "citation_accuracy": {"score": 5, "justification": "Perfect"},
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 100,
                        "cost_usd": 0.001,
                    },
                },
            },
            {
                "example_id": "q2",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "synthesis_prompt": "synthesis.v2",
                "dataset_version": "v2",
                "answer_text": "",
                "refusal": True,
                "cited": [],
                "invalid_citations": [],
                "n_context": 0,
                "gen_usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cost_usd": 0.0,
                },
                "latency_s": {"synthesis": 0.3, "total": 0.3},
                "judge": None,
            },
            {
                "example_id": "q3",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "synthesis_prompt": "synthesis.v2",
                "dataset_version": "v2",
                "answer_text": "Answer to q3",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 2,
                "gen_usage": {
                    "input_tokens": 120,
                    "output_tokens": 60,
                    "cost_usd": 0.0,
                },
                "latency_s": {"synthesis": 0.6, "total": 0.6},
                "judge": {
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-5",
                    "prompt_id": "judge.v1",
                    "faithfulness": {"score": 4, "justification": "Good"},
                    "relevance": {"score": 4, "justification": "Good"},
                    "citation_accuracy": {"score": 4, "justification": "Good"},
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 100,
                        "cost_usd": 0.001,
                    },
                },
            },
        ]

        # Write rows to JSONL
        jsonl_path = results_dir / "ollama--hybrid--none.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        # Summarize
        summary = summarize(results_dir, golden_examples)

        assert summary["run_id"] == Path(tmpdir).name
        assert summary["n_examples"] == 3
        configs = summary["configs"]
        assert isinstance(configs, list)
        assert len(configs) == 1

        cfg_obj = configs[0]
        assert isinstance(cfg_obj, dict)
        cfg = cfg_obj
        assert cfg["provider"] == "ollama"
        assert cfg["n_items"] == 3
        assert cfg["n_judged"] == 2
        assert cfg["n_refusals"] == 1
        assert cfg["n_judge_failures"] == 0
        # q2 is unanswerable and was refused correctly
        assert cfg["refusal_correct_rate"] == 1.0
        # No answerable examples were refused
        assert cfg["false_refusal_rate"] == 0.0

        # Scores should be mean of judged rows (5 and 4)
        assert cfg["scores"]["faithfulness"] == 4.5
        assert cfg["scores"]["relevance"] == 4.5
        assert cfg["scores"]["citation_accuracy"] == 4.5

        # Latency should have mean, p50, p95
        assert "mean" in cfg["latency_s"]
        assert "p50" in cfg["latency_s"]
        assert "p95" in cfg["latency_s"]

        # Gen tokens should sum
        assert cfg["gen_tokens"]["input"] == 270  # 100 + 50 + 120
        assert cfg["gen_tokens"]["output"] == 120  # 50 + 10 + 60

        # Costs should be summed
        assert cfg["gen_cost_usd"] == 0.0
        assert cfg["judge_cost_usd"] == 0.002  # 0.001 * 2 judged


def test_summarize_dataset_version_from_rows_regression(
    golden_examples: list[GoldenExample],
) -> None:
    """Regression: summarize() dataset_version must come from rows, not golden.

    Bug #2: dataset_version was derived from output filename stem ("ollama--hybrid--none"
    → "none") and golden[0].id ("v1" even on v2). Fixed: run_config takes required
    dataset_version kwarg and summarize() reads version from rows.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create row with dataset_version="v2"
        row: dict[str, object] = {
            "example_id": "q1",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "mode": "hybrid",
            "rerank": "none",
            "synthesis_prompt": "synthesis.v2",
            "dataset_version": "v2",
            "answer_text": "test",
            "refusal": False,
            "cited": [],
            "invalid_citations": [],
            "n_context": 1,
            "gen_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
            },
            "latency_s": {"total": 0.1},
            "judge": None,
        }
        jsonl_path = results_dir / "test.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        summary = summarize(results_dir, golden_examples)

        # Summary's dataset_version should come from rows, not golden
        assert summary["dataset_version"] == "v2"


def test_summarize_mixed_versions_regression(
    golden_examples: list[GoldenExample],
) -> None:
    """Regression: summarize dataset_version becomes 'mixed' when rows disagree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create rows with different dataset_versions
        row1: dict[str, object] = {
            "example_id": "q1",
            "provider": "p",
            "model": "m",
            "mode": "hybrid",
            "rerank": "none",
            "synthesis_prompt": "synthesis.v1",
            "dataset_version": "v1",
            "answer_text": "test",
            "refusal": False,
            "cited": [],
            "invalid_citations": [],
            "n_context": 1,
            "gen_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
            },
            "latency_s": {"total": 0.1},
            "judge": None,
        }
        row2: dict[str, object] = {
            "example_id": "q2",
            "provider": "p",
            "model": "m",
            "mode": "hybrid",
            "rerank": "none",
            "synthesis_prompt": "synthesis.v1",
            "dataset_version": "v2",
            "answer_text": "test",
            "refusal": False,
            "cited": [],
            "invalid_citations": [],
            "n_context": 1,
            "gen_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
            },
            "latency_s": {"total": 0.1},
            "judge": None,
        }
        jsonl_path = results_dir / "test.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row1) + "\n")
            f.write(json.dumps(row2) + "\n")

        summary = summarize(results_dir, golden_examples)

        # Should be "mixed" when rows have different versions
        assert summary["dataset_version"] == "mixed"


def test_summarize_ignores_rejudge_regression(
    golden_examples: list[GoldenExample],
) -> None:
    """Regression: summarize() must skip .rejudge files to avoid double-counting.

    Bug #3: summarize() globbed *.jsonl, which double-counts rejudge files
    written into the same run dir. Fixed: it now skips files containing ".rejudge".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create original results file
        original_row: dict[str, object] = {
            "example_id": "q1",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "mode": "hybrid",
            "rerank": "none",
            "synthesis_prompt": "synthesis.v2",
            "dataset_version": "v2",
            "answer_text": "original",
            "refusal": False,
            "cited": [],
            "invalid_citations": [],
            "n_context": 1,
            "gen_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
            },
            "latency_s": {"total": 0.1},
            "judge": None,
        }
        original_path = results_dir / "ollama--hybrid--none.jsonl"
        with original_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(original_row) + "\n")

        # Create rejudge file in the same directory
        rejudge_row: dict[str, object] = {
            "example_id": "q1",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "mode": "hybrid",
            "rerank": "none",
            "synthesis_prompt": "synthesis.v2",
            "dataset_version": "v2",
            "answer_text": "original",
            "refusal": False,
            "cited": [],
            "invalid_citations": [],
            "n_context": 1,
            "gen_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
            },
            "latency_s": {"total": 0.1},
            "judge": {
                "judge_provider": "anthropic",
                "judge_model": "claude-sonnet-5",
                "prompt_id": "judge.v2",
                "faithfulness": {"score": 5, "justification": "Great"},
                "relevance": {"score": 5, "justification": "Great"},
                "citation_accuracy": {"score": 5, "justification": "Great"},
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "cost_usd": 0.001,
                },
            },
        }
        rejudge_path = results_dir / "ollama--hybrid--none.rejudge-anthropic-judge.v2.jsonl"
        with rejudge_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(rejudge_row) + "\n")

        summary = summarize(results_dir, golden_examples)

        # Should have exactly 1 config group with 1 item, not 2
        configs = summary["configs"]
        assert isinstance(configs, list)
        assert len(configs) == 1
        cfg = configs[0]
        assert isinstance(cfg, dict)
        assert cfg["n_items"] == 1  # Only the original, not the rejudge


def test_summarize_sorting() -> None:
    """Test that summarize sorts configs deterministically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create rows in unsorted order
        configs = [
            ("anthropic", "dense", "none"),
            ("ollama", "bm25", "llm"),
            ("anthropic", "bm25", "none"),
            ("google", "hybrid", "cross-encoder"),
        ]

        for provider, mode, rerank in configs:
            jsonl_path = results_dir / f"{provider}--{mode}--{rerank}.jsonl"
            row: dict[str, object] = {
                "example_id": f"ex-{provider}",
                "provider": provider,
                "model": "test-model",
                "mode": mode,
                "rerank": rerank,
                "synthesis_prompt": "synthesis.v1",
                "dataset_version": "v1",
                "answer_text": "test",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 1,
                "gen_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                },
                "latency_s": {"total": 0.1},
                "judge": None,
            }
            with jsonl_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        # Create minimal golden examples
        golden = [
            GoldenExample(
                id=f"ex-{p}",
                question="q",
                reference_answer="a",
                source_citations=[],
                difficulty="easy",
                type="answerable",
            )
            for p, _, _ in configs
        ]

        summary = summarize(results_dir, golden)

        # Should be sorted by (provider, mode, rerank)
        configs_list = summary["configs"]
        assert isinstance(configs_list, list)
        providers_in_order = [
            c["provider"] for c in configs_list if isinstance(c, dict) and "provider" in c
        ]
        assert providers_in_order == ["anthropic", "anthropic", "google", "ollama"]

        # Check modes for anthropic entries
        anthropic_configs = [
            c for c in configs_list if isinstance(c, dict) and c.get("provider") == "anthropic"
        ]
        anthropic_modes: list[Any] = [c.get("mode") for c in anthropic_configs]
        sorted_modes = sorted(anthropic_modes)
        assert anthropic_modes == sorted_modes


def test_eval_set() -> None:
    """Test eval_set drops held-out items and preserves order."""
    golden = [
        GoldenExample(
            id="q1",
            question="q1",
            reference_answer="a1",
            source_citations=[],
            difficulty="easy",
            type="answerable",
            held_out=False,
        ),
        GoldenExample(
            id="q2",
            question="q2",
            reference_answer="a2",
            source_citations=[],
            difficulty="easy",
            type="answerable",
            held_out=True,
        ),
        GoldenExample(
            id="q3",
            question="q3",
            reference_answer="a3",
            source_citations=[],
            difficulty="easy",
            type="answerable",
            held_out=False,
        ),
    ]
    result = eval_set(golden)
    assert len(result) == 2
    assert result[0].id == "q1"
    assert result[1].id == "q3"


async def test_run_config_agentic_pipeline() -> None:
    """Test run_config with agentic pipeline builds AgentMeta."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"

        golden = [
            GoldenExample(
                id="q1",
                question="What is foo?",
                reference_answer="Foo.",
                source_citations=[Citation(doc="doc1", section="sec1")],
                difficulty="easy",
                type="answerable",
                held_out=False,
            ),
        ]

        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            section_id="sec1",
            section_ids=["sec1"],
            section_path="doc1/sec1",
            heading="Section 1",
            page_start=1,
            page_end=1,
            token_count=100,
            text="Text 1",
        )

        # Build a real AgentAnswer for testing
        answer = Answer(
            text="Answer to q1",
            citations=[CitedChunk(marker=1, chunk=chunk)],
            context=[ScoredChunk(chunk=chunk, score=0.9, rank=1)],
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.01),
            timings=[
                StageTiming(stage="planner", seconds=0.1),
                StageTiming(stage="retrieve", seconds=0.2),
                StageTiming(stage="synthesize", seconds=0.1),
                StageTiming(stage="critic", seconds=0.1),
            ],
            refusal=False,
            invalid_citations=[],
        )

        from agentic_rag.agent.state import (
            AgentAnswer,
            CriticVerdict,
            Critique,
            Plan,
            PlanKind,
        )

        agent_answer = AgentAnswer(
            answer=answer,
            plan=Plan(kind=PlanKind.MULTI_HOP, sub_queries=("q1a", "q1b")),
            revisions=1,
            critiques=(Critique(verdict=CriticVerdict.PASS),),
            caveat=False,
            trace=(),
        )

        class AgenticStubPipeline:
            async def ask(self, question: str, **kwargs: Any) -> AgentAnswer:
                return agent_answer

        cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none", pipeline="agentic")
        settings = Settings()

        await run_config(
            cfg,
            golden,
            settings,
            out_path,
            dataset_version="v2",
            concurrency=1,
            do_judge=False,
            _pipeline_factory=lambda s: AgenticStubPipeline(),
        )

        result = json.loads(out_path.read_text().strip())
        assert result["pipeline"] == "agentic"
        assert result["agent"] is not None
        assert result["agent"]["plan_kind"] == "multi_hop"
        assert result["agent"]["sub_queries"] == ["q1a", "q1b"]
        assert result["agent"]["revisions"] == 1
        assert result["agent"]["caveat"] is False


async def test_run_config_agentic_type_check() -> None:
    """Test run_config fails loud when agentic config returns Answer instead of AgentAnswer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.jsonl"

        golden = [
            GoldenExample(
                id="q1",
                question="What is foo?",
                reference_answer="Foo.",
                source_citations=[Citation(doc="doc1", section="sec1")],
                difficulty="easy",
                type="answerable",
                held_out=False,
            ),
        ]

        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            section_id="sec1",
            section_ids=["sec1"],
            section_path="doc1/sec1",
            heading="Section 1",
            page_start=1,
            page_end=1,
            token_count=100,
            text="Text 1",
        )

        # Return vanilla Answer for agentic config (should fail)
        answer = Answer(
            text="Answer to q1",
            citations=[CitedChunk(marker=1, chunk=chunk)],
            context=[ScoredChunk(chunk=chunk, score=0.9, rank=1)],
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.01),
            timings=[StageTiming(stage="synthesis", seconds=0.5)],
            refusal=False,
            invalid_citations=[],
        )

        class VanillaStubPipeline:
            async def ask(self, question: str, **kwargs: Any) -> Answer:
                return answer

        cfg = RunConfig(provider="ollama", mode="hybrid", rerank="none", pipeline="agentic")
        settings = Settings()

        with pytest.raises(TypeError, match="Expected AgentAnswer"):
            await run_config(
                cfg,
                golden,
                settings,
                out_path,
                dataset_version="v2",
                concurrency=1,
                do_judge=False,
                _pipeline_factory=lambda s: VanillaStubPipeline(),
            )


def test_estimate_cost_agentic_higher() -> None:
    """Test that agentic estimate > vanilla estimate for same provider."""
    settings = Settings(
        provider="anthropic",
        anthropic=AnthropicSettings(model="claude-sonnet-5"),
    )

    cfg_vanilla = RunConfig(provider="anthropic", mode="hybrid", rerank="none", pipeline="vanilla")
    cfg_agentic = RunConfig(provider="anthropic", mode="hybrid", rerank="none", pipeline="agentic")

    vanilla_estimates = estimate_cost([cfg_vanilla], n_examples=100, settings=settings)
    agentic_estimates = estimate_cost([cfg_agentic], n_examples=100, settings=settings)

    _, vanilla_cost = vanilla_estimates[0]
    _, agentic_cost = agentic_estimates[0]

    assert vanilla_cost is not None
    assert agentic_cost is not None
    assert agentic_cost > vanilla_cost


def test_summarize_by_type(golden_examples: list[GoldenExample]) -> None:
    """Test summarize computes by_type aggregates correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create rows with different types
        rows: list[dict[str, object]] = [
            {
                "example_id": "q1",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "pipeline": "vanilla",
                "synthesis_prompt": "synthesis.v2",
                "dataset_version": "v2",
                "answer_text": "Answer to q1",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 3,
                "gen_usage": {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.0},
                "latency_s": {"synthesis": 0.5, "total": 0.5},
                "judge": {
                    "judge_provider": "anthropic",
                    "judge_model": "claude-sonnet-5",
                    "prompt_id": "judge.v1",
                    "faithfulness": {"score": 5, "justification": "Perfect"},
                    "relevance": {"score": 5, "justification": "Perfect"},
                    "citation_accuracy": {"score": 5, "justification": "Perfect"},
                    "usage": {"input_tokens": 200, "output_tokens": 100, "cost_usd": 0.001},
                },
            },
            {
                "example_id": "q2",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "pipeline": "vanilla",
                "synthesis_prompt": "synthesis.v2",
                "dataset_version": "v2",
                "answer_text": "",
                "refusal": True,
                "cited": [],
                "invalid_citations": [],
                "n_context": 0,
                "gen_usage": {"input_tokens": 50, "output_tokens": 10, "cost_usd": 0.0},
                "latency_s": {"synthesis": 0.3, "total": 0.3},
                "judge": None,
            },
        ]

        jsonl_path = results_dir / "ollama--hybrid--none--vanilla.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        summary = summarize(results_dir, golden_examples)
        configs = summary["configs"]
        assert len(configs) == 1
        cfg = configs[0]

        # Check by_type exists and has entries for types present
        by_type = cfg.get("by_type", {})
        assert "answerable" in by_type
        assert "unanswerable" in by_type

        # answerable has 1 judged row (q1)
        answerable = by_type["answerable"]
        assert answerable["n"] == 1
        assert answerable["n_judged"] == 1
        assert answerable["scores"]["faithfulness"] == 5.0

        # unanswerable has 1 row (q2, refusal)
        unanswerable = by_type["unanswerable"]
        assert unanswerable["n"] == 1
        assert unanswerable["n_judged"] == 0
        assert unanswerable["refusal_rate"] == 1.0


async def test_summarize_agent_stats(golden_examples: list[GoldenExample]) -> None:
    """Test summarize computes agent stats for agentic configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create rows with agent metadata
        rows: list[dict[str, object]] = [
            {
                "example_id": "q1",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "pipeline": "agentic",
                "synthesis_prompt": "agent-synthesis.v1",
                "dataset_version": "v2",
                "answer_text": "Answer to q1",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 3,
                "gen_usage": {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.0},
                "latency_s": {
                    "planner": 0.1,
                    "retrieve": 0.2,
                    "synthesize": 0.1,
                    "critic": 0.1,
                    "total": 0.5,
                },
                "judge": None,
                "agent": {
                    "plan_kind": "multi_hop",
                    "sub_queries": ["q1a", "q1b"],
                    "revisions": 1,
                    "caveat": False,
                },
            },
            {
                "example_id": "q3",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "mode": "hybrid",
                "rerank": "none",
                "pipeline": "agentic",
                "synthesis_prompt": "agent-synthesis.v1",
                "dataset_version": "v2",
                "answer_text": "Answer to q3",
                "refusal": False,
                "cited": [],
                "invalid_citations": [],
                "n_context": 2,
                "gen_usage": {"input_tokens": 120, "output_tokens": 60, "cost_usd": 0.0},
                "latency_s": {
                    "planner": 0.1,
                    "retrieve": 0.2,
                    "synthesize": 0.2,
                    "critic": 0.2,
                    "total": 0.7,
                },
                "judge": None,
                "agent": {
                    "plan_kind": "direct",
                    "sub_queries": ["q3"],
                    "revisions": 0,
                    "caveat": False,
                },
            },
        ]

        jsonl_path = results_dir / "ollama--hybrid--none--agentic.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        summary = summarize(results_dir, golden_examples)
        configs = summary["configs"]
        assert len(configs) == 1
        cfg = configs[0]

        # Check agent stats
        agent = cfg.get("agent")
        assert agent is not None
        assert agent["multi_hop_rate"] == 0.5  # 1 multi_hop out of 2
        assert agent["mean_revisions"] == 0.5  # (1 + 0) / 2
        assert agent["caveat_rate"] == 0.0  # 0 caveats out of 2
