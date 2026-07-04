"""Tests for synthesis and refusal detection."""

from __future__ import annotations

import pytest

from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.pipeline.synthesize import stream_synthesis, synthesize
from agentic_rag.providers.base import Completion, Message, StreamEvent, Usage
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


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


class FakeLLM:
    """Fake LLM for testing synthesis."""

    def __init__(self, completion_text: str = "", stream_deltas: list[str] | None = None) -> None:
        """Initialize with completion text and optional stream deltas.

        Args:
            completion_text: Text to return for complete() calls.
            stream_deltas: List of text deltas for stream() calls.
        """
        self.completion_text = completion_text
        self.stream_deltas = stream_deltas or []
        self.recorded_messages: list[Message] = []
        self.recorded_model: str | None = None
        self.recorded_max_tokens: int = 0
        self.recorded_temperature: float = 0.0

    @property
    def name(self) -> str:
        return "fake-llm"

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
        self.recorded_model = model or "fake-model"
        self.recorded_max_tokens = max_tokens
        self.recorded_temperature = temperature
        return Completion(
            text=self.completion_text,
            model=self.recorded_model,
            usage=Usage(input_tokens=100, output_tokens=50),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> any:
        """Record call and return streaming generator."""
        self.recorded_messages = list(messages)
        self.recorded_model = model or "fake-model"
        self.recorded_max_tokens = max_tokens
        self.recorded_temperature = temperature
        return self._stream_impl()

    async def _stream_impl(self):
        """Generate stream events from deltas."""
        for delta in self.stream_deltas:
            yield StreamEvent(delta=delta)
        # Terminal event
        yield StreamEvent(
            delta="",
            completion=Completion(
                text=self.completion_text,
                model=self.recorded_model or "fake-model",
                usage=Usage(input_tokens=100, output_tokens=50),
            ),
        )

    def count_tokens(self, text: str) -> int:
        return len(text)


@pytest.mark.asyncio
async def test_normal_completion() -> None:
    """Test normal completion without refusal."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    llm = FakeLLM(completion_text="The answer is here [1].")

    result = await synthesize(llm, "What is this?", context)

    assert result.text == "The answer is here [1]."
    assert result.refusal is False
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.model == "fake-model"
    # Check that prompt was rendered with context and question
    assert len(llm.recorded_messages) == 1
    assert "What is this?" in llm.recorded_messages[0].content
    assert "[1] test" in llm.recorded_messages[0].content


@pytest.mark.asyncio
async def test_refusal_completion() -> None:
    """Test completion with NO_ANSWER_SENTINEL."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    refusal_text = f"{NO_ANSWER_SENTINEL} The corpus lacks this information."
    llm = FakeLLM(completion_text=refusal_text)

    result = await synthesize(llm, "What is this?", context)

    assert result.refusal is True
    assert result.text == "The corpus lacks this information."


@pytest.mark.asyncio
async def test_refusal_with_leading_whitespace() -> None:
    """Test that sentinel is detected even with leading whitespace."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    refusal_text = f"  \n{NO_ANSWER_SENTINEL} No data."
    llm = FakeLLM(completion_text=refusal_text)

    result = await synthesize(llm, "What is this?", context)

    assert result.refusal is True
    assert result.text == "No data."


@pytest.mark.asyncio
async def test_streaming_normal() -> None:
    """Test streaming normal response reassembles correctly."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    deltas = ["The ", "answer ", "[1]."]
    completion_text = "".join(deltas)
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)
        if event.completion:
            # Terminal event
            assert event.delta == ""
            assert event.completion.text == completion_text

    result = "".join(collected_deltas)
    assert result == "The answer [1]."


@pytest.mark.asyncio
async def test_streaming_refusal_split_across_deltas() -> None:
    """Test streaming refusal where sentinel is split across deltas."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Sentinel is "[NO_ANSWER]" - split it across deltas
    deltas = [
        "[NO_",
        "ANSWER]",
        " Not in corpus.",
    ]
    completion_text = "".join(deltas)
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)
        if event.completion:
            # Terminal event is always forwarded
            assert event.completion.text == completion_text

    result = "".join(collected_deltas)
    assert result == " Not in corpus."
    # The sentinel should not appear in the streamed output
    assert NO_ANSWER_SENTINEL not in result


@pytest.mark.asyncio
async def test_streaming_near_miss() -> None:
    """Test streaming where partial sentinel looks like it could be full sentinel."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # "[NO_PE]" looks like it could start with the sentinel but doesn't
    deltas = [
        "[NO_",
        "PE] actually fine",
    ]
    completion_text = "".join(deltas)
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)

    result = "".join(collected_deltas)
    # Since "[NO_PE]" is not the sentinel, all content should be emitted
    assert "[NO_PE]" in result
    assert "actually fine" in result


@pytest.mark.asyncio
async def test_streaming_shorter_than_sentinel() -> None:
    """Test streaming where output is shorter than sentinel prefix."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Just "[NO" which is shorter than full sentinel "[NO_ANSWER]"
    deltas = ["[NO"]
    completion_text = deltas[0]
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)
        if event.completion:
            # Terminal event should be present
            assert event.completion is not None

    result = "".join(collected_deltas)
    assert result == "[NO"


@pytest.mark.asyncio
async def test_streaming_with_custom_model() -> None:
    """Test that custom model is passed through."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    deltas = ["Response"]
    llm = FakeLLM(completion_text="Response", stream_deltas=deltas)

    async for _event in stream_synthesis(llm, "Q?", context, model="custom-model"):
        pass

    assert llm.recorded_model == "custom-model"


@pytest.mark.asyncio
async def test_streaming_empty_delta_after_sentinel() -> None:
    """Test that empty delta after sentinel is emitted."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Sentinel followed by immediate completion
    deltas = [f"{NO_ANSWER_SENTINEL}"]
    completion_text = f"{NO_ANSWER_SENTINEL}"
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)

    result = "".join(collected_deltas)
    # After sentinel is stripped, nothing remains
    assert result == ""
