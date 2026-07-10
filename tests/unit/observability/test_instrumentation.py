"""Span-tree tests: real pipelines under GuardedPipeline, scripted LLM replies.

Each test drives a real RAGPipeline or AgenticPipeline (PlaybackLLM, stub
retriever, NoopReranker) through GuardedPipeline and asserts the exported
span names, nesting, and key attributes — the taxonomy documented in
docs/observability.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentic_rag.agent.graph import AgenticPipeline
from agentic_rag.agent.replay import PlaybackLLM
from agentic_rag.config import Settings
from agentic_rag.guardrails.guarded import GuardedPipeline
from agentic_rag.pipeline.pipeline import RAGPipeline
from agentic_rag.providers.base import Completion, Usage
from agentic_rag.rerank.base import NoopReranker
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="sp800-53r5",
        section_id="AC-2",
        section_ids=["AC-2"],
        section_path="AC-2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        page_start=1,
        page_end=1,
        token_count=8,
        text=text,
    )


def scored(chunk: ChunkRecord, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank)


def completion(text: str) -> Completion:
    return Completion(
        text=text,
        model="playback",
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        stop_reason="end_turn",
    )


class StubRetriever:
    """Returns canned chunks per query string; a default for unknown queries."""

    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.default = [scored(make_chunk("c-ac2", "AC-2 manages accounts."), 1)]

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        return self.results.get(query, self.default)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        guardrails__enabled=True,
        guardrails__audit_enabled=True,
        guardrails__audit_dir=tmp_path / "audit",
    )


def by_name(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def one(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = by_name(spans, name)
    assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)}"
    return matches[0]


def assert_child_of(child: ReadableSpan, parent: ReadableSpan) -> None:
    assert child.parent is not None, f"{child.name} has no parent"
    assert child.context is not None and parent.context is not None
    assert child.parent.span_id == parent.context.span_id, (
        f"{child.name} is not a direct child of {parent.name}"
    )


def guarded_vanilla(settings: Settings, llm: PlaybackLLM) -> GuardedPipeline:
    inner = RAGPipeline(StubRetriever({}), NoopReranker(), llm, settings)
    return GuardedPipeline(inner, settings, provider="playback", model="playback-model")


async def test_vanilla_guarded_ask_span_tree(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    """One vanilla request emits the full tree with correct nesting and attrs."""
    llm = PlaybackLLM.from_completions([completion("AC-2 manages accounts [1].")])
    guarded = guarded_vanilla(settings, llm)

    result = await guarded.ask("What does AC-2 require?")
    assert result.answer.refusal is False

    spans = list(exporter.get_finished_spans())
    assert sorted(s.name for s in spans) == [
        "guardrails.input",
        "guardrails.output",
        "rag.request",
        "rag.rerank",
        "rag.retrieve",
        "rag.synthesize",
    ]

    root = one(spans, "rag.request")
    assert root.parent is None
    assert root.attributes is not None
    assert root.attributes["rag.request_id"] == result.request_id
    assert root.attributes["rag.provider"] == "playback"
    assert root.attributes["rag.model"] == "playback-model"
    assert root.attributes["rag.pipeline"] == "vanilla"
    assert root.attributes["rag.refusal"] is False
    assert isinstance(root.attributes["rag.tokens.input"], int)
    assert isinstance(root.attributes["rag.tokens.output"], int)

    for name in (
        "guardrails.input",
        "rag.retrieve",
        "rag.rerank",
        "rag.synthesize",
        "guardrails.output",
    ):
        assert_child_of(one(spans, name), root)

    guard_in = one(spans, "guardrails.input")
    assert guard_in.attributes is not None
    assert guard_in.attributes["guardrails.blocked"] is False
    assert guard_in.attributes["guardrails.detections"] == 0

    synth = one(spans, "rag.synthesize")
    assert synth.attributes is not None
    assert synth.attributes["rag.tokens.output"] == 50
    assert synth.attributes["rag.citations.invalid_count"] == 0

    rerank = one(spans, "rag.rerank")
    assert rerank.attributes is not None
    assert rerank.attributes["rag.reranker"] == "none"
    assert rerank.attributes["rag.chunks.out"] == 1


async def test_input_blocked_skips_pipeline_spans(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    """A blocked input yields only rag.request + guardrails.input."""
    llm = PlaybackLLM.from_completions([])
    guarded = guarded_vanilla(settings, llm)

    result = await guarded.ask("My SSN is 123-45-6789, what does AC-2 require?")
    assert result.answer.refusal is True

    spans = list(exporter.get_finished_spans())
    assert sorted(s.name for s in spans) == ["guardrails.input", "rag.request"]

    root = one(spans, "rag.request")
    assert root.attributes is not None
    assert root.attributes["rag.refusal"] is True
    assert root.attributes["rag.refusal_reason"] == "input_pii"

    guard_in = one(spans, "guardrails.input")
    assert guard_in.attributes is not None
    assert guard_in.attributes["guardrails.blocked"] is True
    assert_child_of(guard_in, root)


async def test_agentic_guarded_ask_span_tree(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    """An agentic request nests agent.* nodes and per-sub-query retrieval."""
    plan_reply = (
        '{"classification": "multi_hop", '
        '"sub_queries": ["FIPS 199 objectives", "AC-2 requirements"]}'
    )
    llm = PlaybackLLM.from_completions(
        [
            completion(plan_reply),
            completion("FIPS 199 defines objectives [1]. AC-2 manages accounts [2]."),
            completion('{"verdict": "pass"}'),
        ]
    )
    retriever = StubRetriever(
        {
            "FIPS 199 objectives": [scored(make_chunk("c-fips", "FIPS 199 objectives."), 1)],
            "AC-2 requirements": [scored(make_chunk("c-ac2", "AC-2 manages accounts."), 1)],
        }
    )
    inner = AgenticPipeline(retriever, NoopReranker(), llm, settings)
    guarded = GuardedPipeline(inner, settings, provider="playback", model="playback-model")

    result = await guarded.ask("How do FIPS 199 objectives relate to AC-2?")
    assert result.agent is not None

    spans = list(exporter.get_finished_spans())
    root = one(spans, "rag.request")
    assert root.attributes is not None
    assert root.attributes["rag.pipeline"] == "agentic"

    plan = one(spans, "agent.plan")
    gather = one(spans, "agent.gather")
    synthesize = one(spans, "agent.synthesize")
    critic = one(spans, "agent.critic")
    for span in (plan, gather, synthesize, critic):
        assert_child_of(span, root)

    assert plan.attributes is not None
    assert plan.attributes["agent.plan.kind"] == "multi_hop"
    assert plan.attributes["agent.plan.sub_queries"] == 2

    retrieves = by_name(spans, "rag.retrieve")
    assert len(retrieves) == 2
    for span in retrieves:
        assert_child_of(span, gather)
    assert sorted(
        s.attributes["agent.sub_query"] for s in retrieves if s.attributes is not None
    ) == [0, 1]

    assert critic.attributes is not None
    assert critic.attributes["agent.verdict"] == "pass"
    assert critic.attributes["agent.skipped"] is False


async def test_streaming_guarded_ask_span_tree(
    settings: Settings, exporter: InMemorySpanExporter
) -> None:
    """ask_stream emits the same span tree as the non-streaming path."""
    llm = PlaybackLLM.from_completions([completion("AC-2 manages accounts [1].")])
    guarded = guarded_vanilla(settings, llm)

    deltas: list[str] = []
    async for event in guarded.ask_stream("What does AC-2 require?"):
        if event.result is None:
            deltas.append(event.delta)
    assert "".join(deltas)

    spans = list(exporter.get_finished_spans())
    assert sorted(s.name for s in spans) == [
        "guardrails.input",
        "guardrails.output",
        "rag.request",
        "rag.rerank",
        "rag.retrieve",
        "rag.synthesize",
    ]
    root = one(spans, "rag.request")
    for name in ("guardrails.input", "rag.retrieve", "rag.rerank", "guardrails.output"):
        assert_child_of(one(spans, name), root)
