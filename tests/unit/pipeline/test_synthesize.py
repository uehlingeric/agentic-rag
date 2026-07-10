"""Tests for synthesis and refusal detection."""

from __future__ import annotations

import pytest

from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL, scrub_sentinel
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
    assert result.stray_sentinel is False
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


@pytest.mark.asyncio
async def test_mid_answer_sentinel() -> None:
    """Test mid-answer sentinel: partial claim, then sentinel, then caveat."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Mid-answer: starts with normal text, has sentinel in middle
    completion_text = "Partial claim [1].\n\n[NO_ANSWER] The excerpts do not state X."
    llm = FakeLLM(completion_text=completion_text)

    result = await synthesize(llm, "What is X?", context)

    # Should NOT be a refusal (sentinel is not leading)
    assert result.refusal is False
    # Should mark stray sentinel
    assert result.stray_sentinel is True
    # Text should keep both claim and caveat, no sentinel
    assert "Partial claim [1]." in result.text
    assert "The excerpts do not state X." in result.text
    assert NO_ANSWER_SENTINEL not in result.text
    # Paragraph break preserved since newline follows period
    assert ".\n\nThe" in result.text


@pytest.mark.asyncio
async def test_trailing_sentinel_directly_at_end() -> None:
    """Test sentinel directly at end of answer."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    completion_text = f"The answer is here [1]. {NO_ANSWER_SENTINEL}"
    llm = FakeLLM(completion_text=completion_text)

    result = await synthesize(llm, "What is this?", context)

    assert result.refusal is False
    assert result.stray_sentinel is True
    assert result.text == "The answer is here [1]."
    assert NO_ANSWER_SENTINEL not in result.text


@pytest.mark.asyncio
async def test_inline_mid_sentence_sentinel() -> None:
    """Test sentinel inline mid-sentence."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Sentinel inserted in middle of a sentence
    completion_text = "maintaining them [NO_ANSWER] The excerpts do not state Y."
    llm = FakeLLM(completion_text=completion_text)

    result = await synthesize(llm, "What?", context)

    assert result.refusal is False
    assert result.stray_sentinel is True
    # Text should be joined with single space
    assert result.text == "maintaining them The excerpts do not state Y."
    assert NO_ANSWER_SENTINEL not in result.text


@pytest.mark.asyncio
async def test_leading_and_stray_sentinel() -> None:
    """Test leading sentinel + stray occurrence later."""
    chunk = make_chunk("c1", text="Test data.")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Leading sentinel means refusal, but there's also a stray one
    completion_text = f"{NO_ANSWER_SENTINEL} Some text [NO_ANSWER] more text."
    llm = FakeLLM(completion_text=completion_text)

    result = await synthesize(llm, "What?", context)

    # Leading sentinel takes priority: this is a refusal
    assert result.refusal is True
    # But stray occurrence was also found and removed
    assert result.stray_sentinel is True
    # Text after leading sentinel has stray removed
    assert result.text == "Some text more text."
    assert NO_ANSWER_SENTINEL not in result.text


@pytest.mark.asyncio
async def test_streaming_mid_answer_sentinel() -> None:
    """Test streaming with mid-answer sentinel in terminal completion."""
    chunk = make_chunk("c1")
    scored = ScoredChunk(chunk=chunk, score=0.9, rank=1)
    context = BuiltContext(text="[1] test\n", chunks=[scored], token_count=10)

    # Streamed deltas are clean, but terminal completion has stray sentinel
    deltas = ["The answer is ", "here [1]."]
    completion_text = "The answer is here [1]. [NO_ANSWER] But the context is limited."
    llm = FakeLLM(completion_text=completion_text, stream_deltas=deltas)

    collected_deltas = []
    terminal_completion = None
    async for event in stream_synthesis(llm, "Q?", context):
        if event.delta:
            collected_deltas.append(event.delta)
        if event.completion:
            terminal_completion = event.completion

    # Streamed deltas should NOT include sentinel (streaming only buffers leading)
    streamed = "".join(collected_deltas)
    assert NO_ANSWER_SENTINEL not in streamed

    # Terminal completion still has the raw text (not post-processed by stream_synthesis)
    assert NO_ANSWER_SENTINEL in terminal_completion.text
    assert "The answer is here" in terminal_completion.text


def test_scrub_sentinel_empty_string() -> None:
    """Test scrub_sentinel with empty string."""
    result = scrub_sentinel("")
    assert result.text == ""
    assert result.refusal is False
    assert result.stray_sentinel is False


def test_scrub_sentinel_only_sentinel() -> None:
    """Test scrub_sentinel with only sentinel."""
    result = scrub_sentinel(f"{NO_ANSWER_SENTINEL}")
    assert result.text == ""
    assert result.refusal is True
    assert result.stray_sentinel is False


def test_scrub_sentinel_leading_only() -> None:
    """Test scrub_sentinel with leading sentinel and text."""
    result = scrub_sentinel(f"{NO_ANSWER_SENTINEL} The answer.")
    assert result.text == "The answer."
    assert result.refusal is True
    assert result.stray_sentinel is False


def test_scrub_sentinel_stray_only() -> None:
    """Test scrub_sentinel with stray (non-leading) sentinel."""
    result = scrub_sentinel(f"The answer. {NO_ANSWER_SENTINEL}")
    assert result.text == "The answer."
    assert result.refusal is False
    assert result.stray_sentinel is True


def test_scrub_sentinel_with_whitespace() -> None:
    """Test scrub_sentinel with leading/trailing whitespace."""
    result = scrub_sentinel(f"  \n{NO_ANSWER_SENTINEL} Answer text.  ")
    assert result.text == "Answer text."
    assert result.refusal is True
    assert result.stray_sentinel is False
