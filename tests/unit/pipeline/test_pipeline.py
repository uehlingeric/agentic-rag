"""Tests for RAG pipeline orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace

from agentic_rag.config import Settings
from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL
from agentic_rag.pipeline.pipeline import RAGPipeline
from agentic_rag.providers.base import Completion, Message, StreamEvent, Usage
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "test",
    section_id: str = "SEC",
    heading: str = "Test Section",
    text: str = "Test content.",
) -> ChunkRecord:
    """Create a test chunk."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=heading,
        heading=heading,
        page_start=1,
        page_end=1,
        token_count=10,
        text=text,
    )


def make_scored(chunk: ChunkRecord, score: float = 0.9, rank: int = 1) -> ScoredChunk:
    """Create a scored chunk."""
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


class StubRetriever:
    """Test retriever that records calls and returns preset results."""

    def __init__(self, results: list[ScoredChunk] | None = None) -> None:
        """Initialize with preset results."""
        self.results = results or []
        self.last_query: str | None = None
        self.last_mode: RetrievalMode | None = None
        self.last_top_k: int | None = None

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Record call and return results."""
        self.last_query = query
        self.last_mode = mode
        self.last_top_k = top_k
        return self.results


class StubReranker:
    """Test reranker that optionally reorders and returns preset usage."""

    def __init__(self, reorder: bool = False) -> None:
        """Initialize reranker.

        Args:
            reorder: If True, reverses the candidate order before cutting.
        """
        self.name = "stub-rerank"
        self.last_usage = Usage.zero()
        self.reorder = reorder
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        """Rerank and return top_k results."""
        self.last_query = query
        self.last_top_k = top_k

        # Optionally reorder
        result = list(candidates[:top_k])
        if self.reorder:
            result = list(reversed(result))

        # Reassign ranks
        return [replace(c, rank=i) for i, c in enumerate(result, start=1)]


class StubLLM:
    """Test LLM for completion and streaming."""

    def __init__(
        self,
        completion_text: str = "",
        stream_deltas: list[str] | None = None,
        usage: Usage | None = None,
        custom_count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize with completion text and stream deltas.

        Args:
            completion_text: Text returned by complete() and stream().
            stream_deltas: List of text deltas for streaming.
            usage: Usage to return; defaults to fixed value.
            custom_count_tokens: Optional custom token counter.
        """
        self.name = "stub-llm"
        self.completion_text = completion_text
        self.stream_deltas = stream_deltas or []
        self.usage = usage or Usage(input_tokens=100, output_tokens=50, cost_usd=0.0)
        self.custom_count_tokens = custom_count_tokens
        self.recorded_messages: list[Message] = []
        self.recorded_model: str | None = None
        self.recorded_max_tokens: int = 0
        self.recorded_temperature: float = 0.0

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Record call and return completion."""
        self.recorded_messages = list(messages)
        self.recorded_model = model or "stub-model"
        self.recorded_max_tokens = max_tokens
        self.recorded_temperature = temperature
        return Completion(
            text=self.completion_text,
            model=self.recorded_model,
            usage=self.usage,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Record call and return streaming generator."""
        self.recorded_messages = list(messages)
        self.recorded_model = model or "stub-model"
        self.recorded_max_tokens = max_tokens
        self.recorded_temperature = temperature
        return self._stream_impl()

    async def _stream_impl(self) -> AsyncIterator[StreamEvent]:
        """Generate stream events from deltas."""
        for delta in self.stream_deltas:
            yield StreamEvent(delta=delta)
        # Terminal event
        yield StreamEvent(
            delta="",
            completion=Completion(
                text=self.completion_text,
                model=self.recorded_model or "stub-model",
                usage=self.usage,
            ),
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.custom_count_tokens:
            return self.custom_count_tokens(text)
        # Default: whitespace-split word count
        return len(text.split())


async def test_happy_path(settings: Settings) -> None:
    """Test normal completion with citations."""
    # Setup: 3 chunks
    chunks = [
        make_chunk("c1", text="First fact about topic"),
        make_chunk("c2", text="Second fact about topic"),
        make_chunk("c3", text="Third fact about topic"),
    ]
    scored = [make_scored(c, score=0.9 - i * 0.1, rank=i + 1) for i, c in enumerate(chunks)]

    # Create stubs
    retriever = StubRetriever(results=scored)
    reranker = StubReranker()
    llm = StubLLM(completion_text="First fact [1]. Second fact [3].")
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    # Call ask
    answer = await pipeline.ask("What are the facts?")

    # Assertions: citations should resolve to chunks 1 and 3
    assert len(answer.citations) == 2
    assert answer.citations[0].marker == 1
    assert answer.citations[0].chunk.chunk_id == "c1"
    assert answer.citations[1].marker == 3
    assert answer.citations[1].chunk.chunk_id == "c3"

    # Text should have markers
    assert "[1]" in answer.text
    assert "[3]" in answer.text

    # No refusal, no invalid citations
    assert answer.refusal is False
    assert answer.invalid_citations == []

    # Context is the built chunk list (reranked, top_k=8)
    assert len(answer.context) == 3
    assert answer.context[0].chunk.chunk_id == "c1"

    # Timings should be 3 entries
    assert len(answer.timings) == 3
    assert answer.timings[0].stage == "retrieve"
    assert answer.timings[1].stage == "rerank"
    assert answer.timings[2].stage == "synthesize"
    for timing in answer.timings:
        assert timing.seconds >= 0.0


async def test_plumbing(settings: Settings) -> None:
    """Test that retriever and reranker are called with correct parameters."""
    chunks = [make_scored(make_chunk(f"c{i}"), rank=i + 1) for i in range(5)]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()
    llm = StubLLM(completion_text="Answer [1].")
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    # Call with specific mode
    await pipeline.ask("Q?", mode=RetrievalMode.BM25)

    # Retriever should see candidate_pool
    assert retriever.last_query == "Q?"
    assert retriever.last_mode == RetrievalMode.BM25
    assert retriever.last_top_k == settings.rerank.candidate_pool  # 30

    # Reranker should see top_k
    assert reranker.last_query == "Q?"
    assert reranker.last_top_k == settings.rerank.top_k  # 8


async def test_invalid_marker(settings: Settings) -> None:
    """Test that invalid citations are stripped and recorded."""
    chunks = [
        make_scored(make_chunk("c1", text="First")),
        make_scored(make_chunk("c2", text="Second")),
        make_scored(make_chunk("c3", text="Third")),
    ]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()
    # LLM tries to cite chunk 7 which doesn't exist (only 3 in context)
    llm = StubLLM(completion_text="First chunk [1]. Missing chunk [7]. Third chunk [3].")
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    answer = await pipeline.ask("Q?")

    # Invalid marker 7 should be stripped from text
    assert "[7]" not in answer.text
    assert "First chunk [1]" in answer.text
    assert "Third chunk [3]" in answer.text

    # Invalid marker 7 should be recorded
    assert answer.invalid_citations == [7]

    # Valid citations should still resolve
    assert len(answer.citations) == 2
    assert answer.citations[0].marker == 1
    assert answer.citations[1].marker == 3


async def test_refusal(settings: Settings) -> None:
    """Test handling of refusal sentinel."""
    chunks = [make_scored(make_chunk("c1"))]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()
    llm = StubLLM(completion_text=f"{NO_ANSWER_SENTINEL}")
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    answer = await pipeline.ask("Q?")

    # Refusal should be set
    assert answer.refusal is True
    # Text should be empty (sentinel stripped)
    assert answer.text == ""
    # No citations
    assert answer.citations == []


async def test_usage_summation(settings: Settings) -> None:
    """Test that usage is summed from reranker and synthesis."""
    chunks = [make_scored(make_chunk("c1"))]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()
    reranker.last_usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.0)

    synth_usage = Usage(input_tokens=200, output_tokens=80, cost_usd=0.0)
    llm = StubLLM(completion_text="Answer.", usage=synth_usage)
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    answer = await pipeline.ask("Q?")

    # Usage should be summed
    assert answer.usage.input_tokens == 300  # 100 + 200
    assert answer.usage.output_tokens == 130  # 50 + 80
    assert answer.usage.cost_usd == 0.0


async def test_token_budget_edge(settings: Settings) -> None:
    """Test that only first chunk fits in context token budget."""
    chunks = [
        make_scored(make_chunk("c1", text="x" * 100)),
        make_scored(make_chunk("c2", text="y" * 100)),
        make_scored(make_chunk("c3", text="z" * 100)),
    ]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()

    # Custom token counter: first excerpt fits, second doesn't
    call_count = {"count": 0}

    def custom_count(text: str) -> int:
        call_count["count"] += 1
        # Each individual excerpt counts as 3500 tokens
        # First excerpt alone: 3500 (fits in 6000 budget)
        # First + second: 7000 (exceeds 6000 budget, so second is dropped)
        if text.count("[") == 1:
            return 3500  # Single excerpt
        # Multiple excerpts combined
        return call_count["count"] * 3500

    llm = StubLLM(completion_text="Only first [1].", custom_count_tokens=custom_count)
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    answer = await pipeline.ask("Q?")

    # Only first chunk should be in context
    assert len(answer.context) == 1
    assert answer.context[0].chunk.chunk_id == "c1"


async def test_streaming_normal_path(settings: Settings) -> None:
    """Test streaming normal response with text deltas."""
    chunks = [make_scored(make_chunk("c1"))]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()

    deltas = ["First ", "fact ", "[1]. ", "Second ", "[1]."]
    completion_text = "".join(deltas)
    llm = StubLLM(completion_text=completion_text, stream_deltas=deltas)
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    # Collect events
    text_events = []
    terminal_event = None
    async for event in pipeline.ask_stream("Q?"):
        if event.delta:
            text_events.append(event)
        if event.answer is not None:
            terminal_event = event

    # Should have text deltas
    assert len(text_events) > 0
    assert "".join(e.delta for e in text_events) == completion_text

    # Should have exactly one terminal event
    assert terminal_event is not None
    assert terminal_event.answer is not None
    assert terminal_event.delta == ""
    assert terminal_event.answer.text == completion_text
    assert terminal_event.answer.refusal is False

    # Answer should have citations resolved
    assert len(terminal_event.answer.citations) == 1
    assert terminal_event.answer.citations[0].marker == 1
    assert len(terminal_event.answer.timings) == 3


async def test_streaming_refusal_split_sentinel(settings: Settings) -> None:
    """Test streaming refusal where sentinel is split across deltas."""
    chunks = [make_scored(make_chunk("c1"))]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()

    # Sentinel "[NO_ANSWER]" split across deltas
    deltas = ["[NO_", "ANSWER]"]
    completion_text = "".join(deltas)
    llm = StubLLM(completion_text=completion_text, stream_deltas=deltas)
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    # Collect events
    text_events = []
    terminal_event = None
    async for event in pipeline.ask_stream("Q?"):
        if event.delta:
            text_events.append(event)
        if event.answer is not None:
            terminal_event = event

    # No text deltas should be emitted (sentinel buffered)
    assert len(text_events) == 0

    # Terminal event should show refusal
    assert terminal_event is not None
    assert terminal_event.answer is not None
    assert terminal_event.answer.refusal is True
    assert terminal_event.answer.text == ""


async def test_streaming_terminal_answer_usage(settings: Settings) -> None:
    """Test that terminal answer has correct usage summation."""
    chunks = [make_scored(make_chunk("c1"))]
    retriever = StubRetriever(results=chunks)
    reranker = StubReranker()
    reranker.last_usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.0)

    synth_usage = Usage(input_tokens=200, output_tokens=80, cost_usd=0.0)
    llm = StubLLM(
        completion_text="Streamed answer.",
        stream_deltas=["Streamed ", "answer."],
        usage=synth_usage,
    )
    pipeline = RAGPipeline(retriever, reranker, llm, settings)

    terminal_event = None
    async for event in pipeline.ask_stream("Q?"):
        if event.answer is not None:
            terminal_event = event

    assert terminal_event is not None
    assert terminal_event.answer is not None
    # Usage should be summed from reranker + synthesis
    assert terminal_event.answer.usage.input_tokens == 300  # 100 + 200
    assert terminal_event.answer.usage.output_tokens == 130  # 50 + 80
