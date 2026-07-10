"""Tests for metrics store: record, summary, aggregation, and formatting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentic_rag.config import MetricsSettings, Settings
from agentic_rag.guardrails.guarded import GuardedPipeline
from agentic_rag.observability.metrics import MetricsStore, RequestMetric, metrics_store_for
from agentic_rag.pipeline.base import Answer, StageTiming
from agentic_rag.pipeline.pipeline import AskStreamEvent
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


@pytest.fixture
def metrics_db(tmp_path: Path) -> Path:
    """Return a path to a fresh metrics database."""
    return tmp_path / "metrics.db"


def make_chunk(chunk_id: str) -> ChunkRecord:
    """Create a test chunk."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="test",
        section_id="SEC",
        section_ids=["SEC"],
        section_path="Test Section",
        heading="Test",
        page_start=1,
        page_end=1,
        token_count=10,
        text="Test content.",
    )


def make_scored(chunk: ChunkRecord) -> ScoredChunk:
    """Create a scored chunk."""
    return ScoredChunk(chunk=chunk, score=0.9, rank=1)


def make_answer(text: str = "Test answer") -> Answer:
    """Create a test Answer."""
    return Answer(
        text=text,
        citations=[],
        context=[],
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.123),
        timings=[
            StageTiming("retrieve", 0.1),
            StageTiming("synthesize", 0.2),
        ],
        refusal=False,
        refusal_reason=None,
    )


class TestMetricsStore:
    """Test MetricsStore basic operations."""

    def test_store_creates_db_and_tables(self, metrics_db: Path) -> None:
        """Store initializes database with correct schema."""
        MetricsStore(metrics_db)
        assert metrics_db.exists()

    def test_record_one_metric(self, metrics_db: Path) -> None:
        """Record a single metric."""
        store = MetricsStore(metrics_db)
        metric = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="claude-sonnet-5",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.123,
            latency_s=1.5,
            stages={"retrieve": 0.5, "synthesize": 1.0},
            refusal=False,
            refusal_reason=None,
        )
        store.record(metric)

        # Verify by querying the database directly
        import sqlite3

        with sqlite3.connect(metrics_db) as conn:
            row = conn.execute(
                "SELECT request_id, provider, input_tokens FROM metrics WHERE request_id = ?",
                ("req-1",),
            ).fetchone()
            assert row is not None
            assert row[0] == "req-1"
            assert row[1] == "anthropic"
            assert row[2] == 100

    def test_record_upsert_by_request_id_source(self, metrics_db: Path) -> None:
        """Duplicate (request_id, source) replaces previous row."""
        store = MetricsStore(metrics_db)
        metric1 = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="claude-sonnet-5",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.123,
            latency_s=1.5,
            stages={},
            refusal=False,
            refusal_reason=None,
        )
        store.record(metric1)

        # Record again with same (request_id, source) but different tokens
        metric2 = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="claude-sonnet-5",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=200,  # Changed
            output_tokens=100,
            cost_usd=0.250,
            latency_s=3.0,
            stages={},
            refusal=False,
            refusal_reason=None,
        )
        store.record(metric2)

        # Should have only one row
        import sqlite3

        with sqlite3.connect(metrics_db) as conn:
            rows = conn.execute(
                "SELECT input_tokens FROM metrics WHERE request_id = ?", ("req-1",)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] == 200

    def test_summary_groups_by_provider(self, metrics_db: Path) -> None:
        """Summary aggregates correctly by provider."""
        store = MetricsStore(metrics_db)
        metrics = [
            RequestMetric(
                request_id=f"req-{i}",
                ts=f"2025-01-15T10:0{i}:00+00:00",
                source="cli",
                provider=provider,
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=1.0,
                latency_s=1.0 + i * 0.1,
                stages={},
                refusal=i == 2,  # One refusal
                refusal_reason="input_pii" if i == 2 else None,
            )
            for i, provider in enumerate(["anthropic", "anthropic", "google"])
        ]
        for m in metrics:
            store.record(m)

        rows = store.summary(group="provider")
        assert len(rows) == 2

        anthropic_row = next(r for r in rows if r["group"] == "anthropic")
        assert anthropic_row["requests"] == 2
        assert anthropic_row["refusals"] == 0
        assert anthropic_row["input_tokens"] == 200
        assert anthropic_row["output_tokens"] == 100
        assert anthropic_row["cost_usd"] == 2.0

        google_row = next(r for r in rows if r["group"] == "google")
        assert google_row["refusals"] == 1

    def test_summary_day_grouping(self, metrics_db: Path) -> None:
        """Summary groups correctly by day."""
        store = MetricsStore(metrics_db)
        metrics = [
            RequestMetric(
                request_id=f"req-{i}",
                ts=ts,
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=1.0,
                latency_s=1.0,
                stages={},
                refusal=False,
                refusal_reason=None,
            )
            for i, ts in enumerate(
                [
                    "2025-01-15T10:00:00+00:00",
                    "2025-01-15T11:00:00+00:00",
                    "2025-01-16T10:00:00+00:00",
                ]
            )
        ]
        for m in metrics:
            store.record(m)

        rows = store.summary(group="day")
        assert len(rows) == 2
        day_15 = next(r for r in rows if r["group"] == "2025-01-15")
        assert day_15["requests"] == 2
        day_16 = next(r for r in rows if r["group"] == "2025-01-16")
        assert day_16["requests"] == 1

    def test_summary_since_filter(self, metrics_db: Path) -> None:
        """Summary filters by since date."""
        store = MetricsStore(metrics_db)
        metrics = [
            RequestMetric(
                request_id=f"req-{i}",
                ts=ts,
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=1.0,
                latency_s=1.0,
                stages={},
                refusal=False,
                refusal_reason=None,
            )
            for i, ts in enumerate(
                [
                    "2025-01-14T10:00:00+00:00",
                    "2025-01-15T10:00:00+00:00",
                    "2025-01-16T10:00:00+00:00",
                ]
            )
        ]
        for m in metrics:
            store.record(m)

        rows = store.summary(since="2025-01-15")
        total_requests = sum(r["requests"] for r in rows)
        assert total_requests == 2  # Only 15th and 16th

    def test_summary_cost_null_handling(self, metrics_db: Path) -> None:
        """Cost is None when all costs are NULL, sum when any are present."""
        store = MetricsStore(metrics_db)
        metrics = [
            RequestMetric(
                request_id="req-1",
                ts="2025-01-15T10:00:00+00:00",
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=None,
                latency_s=1.0,
                stages={},
                refusal=False,
                refusal_reason=None,
            ),
            RequestMetric(
                request_id="req-2",
                ts="2025-01-15T10:00:00+00:00",
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=None,
                latency_s=1.0,
                stages={},
                refusal=False,
                refusal_reason=None,
            ),
        ]
        for m in metrics:
            store.record(m)

        rows = store.summary(group="provider")
        assert rows[0]["cost_usd"] is None

        # Now add one with cost
        m3 = RequestMetric(
            request_id="req-3",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="model",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.5,
            latency_s=1.0,
            stages={},
            refusal=False,
            refusal_reason=None,
        )
        store.record(m3)

        rows = store.summary(group="provider")
        assert rows[0]["cost_usd"] == 0.5  # Sum of non-null

    def test_summary_percentiles_nearest_rank(self, metrics_db: Path) -> None:
        """Latency percentiles use nearest-rank method."""
        store = MetricsStore(metrics_db)
        # Create metrics with known latencies: 1, 2, ..., 10
        for i in range(1, 11):
            m = RequestMetric(
                request_id=f"req-{i}",
                ts="2025-01-15T10:00:00+00:00",
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=1.0,
                latency_s=float(i),  # 1.0, 2.0, ..., 10.0
                stages={},
                refusal=False,
                refusal_reason=None,
            )
            store.record(m)

        rows = store.summary(group="provider")
        row = rows[0]
        # p50 at rank ceil(0.5 * 10) - 1 = 4 (0-indexed) => 5.0
        # p95 at rank ceil(0.95 * 10) - 1 = 9 (0-indexed) => 10.0
        assert row["latency_p50"] == 5.0
        assert row["latency_p95"] == 10.0

    def test_stage_summary_parses_json(self, metrics_db: Path) -> None:
        """Stage summary parses stages JSON correctly."""
        store = MetricsStore(metrics_db)
        m1 = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="model",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=1.0,
            latency_s=1.5,
            stages={"retrieve": 0.5, "synthesize": 1.0},
            refusal=False,
            refusal_reason=None,
        )
        m2 = RequestMetric(
            request_id="req-2",
            ts="2025-01-15T10:01:00+00:00",
            source="cli",
            provider="anthropic",
            model="model",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=1.0,
            latency_s=2.0,
            stages={"retrieve": 0.8, "synthesize": 1.2},
            refusal=False,
            refusal_reason=None,
        )
        store.record(m1)
        store.record(m2)

        rows = store.stage_summary()
        assert len(rows) == 2
        retrieve = next(r for r in rows if r["stage"] == "retrieve")
        assert retrieve["count"] == 2
        # p50 with 2 values: rank = ceil(0.5 * 2) - 1 = 1 - 1 = 0, so 0.5
        assert retrieve["p50"] == 0.5

    def test_format_summary_returns_markdown_table(self, metrics_db: Path) -> None:
        """format_summary returns a markdown table string."""
        store = MetricsStore(metrics_db)
        m = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="model",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.123,
            latency_s=1.5,
            stages={},
            refusal=False,
            refusal_reason=None,
        )
        store.record(m)
        rows = store.summary(group="provider")

        table = store.format_summary(rows, group="provider")
        assert "| Provider |" in table or "| provider |" in table
        assert "1,000" in table  # Thousands separator
        assert "0.1230" in table  # Cost formatted

    def test_format_stages_returns_markdown_table(self, metrics_db: Path) -> None:
        """format_stages returns a markdown table string."""
        store = MetricsStore(metrics_db)
        m = RequestMetric(
            request_id="req-1",
            ts="2025-01-15T10:00:00+00:00",
            source="cli",
            provider="anthropic",
            model="model",
            pipeline="vanilla",
            mode="hybrid",
            rerank="none",
            input_tokens=100,
            output_tokens=50,
            cost_usd=1.0,
            latency_s=1.5,
            stages={"retrieve": 0.5, "synthesize": 1.0},
            refusal=False,
            refusal_reason=None,
        )
        store.record(m)
        rows = store.stage_summary()

        table = store.format_stages(rows)
        assert "| Stage |" in table or "| stage |" in table
        assert "retrieve" in table
        assert "synthesize" in table

    def test_bad_group_raises_valueerror(self, metrics_db: Path) -> None:
        """Invalid group parameter raises ValueError."""
        store = MetricsStore(metrics_db)
        with pytest.raises(ValueError, match="Invalid group"):
            store.summary(group="invalid")

    def test_metrics_store_for_enabled(self, tmp_path: Path) -> None:
        """metrics_store_for returns store when enabled."""
        settings = Settings(
            data_dir=tmp_path / "data",
            metrics=MetricsSettings(enabled=True),
        )
        store = metrics_store_for(settings)
        assert store is not None
        assert isinstance(store, MetricsStore)

    def test_metrics_store_for_disabled(self, tmp_path: Path) -> None:
        """metrics_store_for returns None when disabled."""
        settings = Settings(
            data_dir=tmp_path / "data",
            metrics=MetricsSettings(enabled=False),
        )
        store = metrics_store_for(settings)
        assert store is None

    def test_metrics_store_for_custom_path(self, tmp_path: Path) -> None:
        """metrics_store_for uses custom db_path when set."""
        custom_db = tmp_path / "custom.db"
        settings = Settings(
            data_dir=tmp_path / "data",
            metrics=MetricsSettings(
                enabled=True,
                db_path=custom_db,
            ),
        )
        store = metrics_store_for(settings)
        assert store is not None
        store.record(
            RequestMetric(
                request_id="req-1",
                ts="2025-01-15T10:00:00+00:00",
                source="cli",
                provider="anthropic",
                model="model",
                pipeline="vanilla",
                mode="hybrid",
                rerank="none",
                input_tokens=100,
                output_tokens=50,
                cost_usd=1.0,
                latency_s=1.5,
                stages={},
                refusal=False,
                refusal_reason=None,
            )
        )
        assert custom_db.exists()


class StubVanillaWithSettings:
    """Stub vanilla pipeline for GuardedPipeline testing."""

    def __init__(self, answer: Answer | None = None) -> None:
        self.answer = answer or make_answer()
        self.settings = Settings()

    async def ask(self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID) -> Answer:
        return self.answer

    def ask_stream(
        self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> AsyncIterator[AskStreamEvent]:
        async def _stream() -> AsyncIterator[AskStreamEvent]:
            yield AskStreamEvent(delta=self.answer.text)
            yield AskStreamEvent(answer=self.answer)

        return _stream()


class TestGuardedPipelineMetrics:
    """Test metrics recording in GuardedPipeline."""

    async def test_ask_records_metric_on_success(self, tmp_path: Path) -> None:
        """GuardedPipeline.ask() records a metric on successful completion."""
        inner = StubVanillaWithSettings(make_answer("Test answer"))
        settings = Settings(
            data_dir=tmp_path / "data",
            guardrails__enabled=True,
            guardrails__audit_enabled=False,
            metrics__enabled=True,
            metrics__db_path=tmp_path / "metrics.db",
        )
        inner.settings = settings
        guarded = GuardedPipeline(inner, settings, provider="test", model="test-model")

        await guarded.ask("Clean question")

        # Verify metric was recorded
        assert guarded.metrics is not None
        rows = guarded.metrics.summary(group="provider")
        assert len(rows) == 1
        assert rows[0]["requests"] == 1
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["output_tokens"] == 50

    async def test_ask_records_metric_on_input_blocked(self, tmp_path: Path) -> None:
        """GuardedPipeline.ask() records refusal metric when input is blocked."""
        inner = StubVanillaWithSettings()
        settings = Settings(
            data_dir=tmp_path / "data",
            guardrails__enabled=True,
            guardrails__audit_enabled=False,
            metrics__enabled=True,
            metrics__db_path=tmp_path / "metrics.db",
        )
        inner.settings = settings
        guarded = GuardedPipeline(inner, settings, provider="test", model="test-model")

        await guarded.ask("What is my SSN 123-45-6789?")

        # Verify metric was recorded with refusal
        assert guarded.metrics is not None
        rows = guarded.metrics.summary(group="provider")
        assert len(rows) == 1
        assert rows[0]["refusals"] == 1
        assert rows[0]["input_tokens"] == 0
        assert rows[0]["output_tokens"] == 0

    async def test_ask_metrics_disabled(self, tmp_path: Path) -> None:
        """GuardedPipeline with metrics disabled creates no database."""
        inner = StubVanillaWithSettings()
        settings = Settings(
            data_dir=tmp_path / "data",
            guardrails__enabled=True,
            guardrails__audit_enabled=False,
            metrics=MetricsSettings(enabled=False),
        )
        inner.settings = settings
        guarded = GuardedPipeline(inner, settings, provider="test", model="test-model")

        await guarded.ask("Test question")

        # Verify metrics store was not created
        assert guarded.metrics is None
        # No metrics db should exist
        assert not (tmp_path / "data" / "metrics.db").exists()

    async def test_ask_stream_records_metric(self, tmp_path: Path) -> None:
        """GuardedPipeline.ask_stream() records one metric at the terminal event."""
        inner = StubVanillaWithSettings(make_answer("Streamed answer"))
        settings = Settings(
            data_dir=tmp_path / "data",
            guardrails__enabled=True,
            guardrails__audit_enabled=False,
            metrics__enabled=True,
            metrics__db_path=tmp_path / "metrics.db",
        )
        inner.settings = settings
        guarded = GuardedPipeline(inner, settings, provider="test", model="test-model")

        async for _event in guarded.ask_stream("Clean question"):
            pass

        assert guarded.metrics is not None
        rows = guarded.metrics.summary(group="source")
        assert len(rows) == 1
        assert rows[0]["group"] == "cli"
        assert rows[0]["requests"] == 1
        assert rows[0]["input_tokens"] == 100
