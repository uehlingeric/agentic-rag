"""Tests for recording and replaying LLM provider calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from agentic_rag.agent.replay import (
    PlaybackLLM,
    RecordedCall,
    RecordingLLM,
    ReplayExhaustedError,
    load_cassette,
    save_cassette,
)
from agentic_rag.providers.base import Completion, Message, Role, StreamEvent, Usage


class StubLLM:
    """Minimal test double for LLMProvider."""

    def __init__(self, replies: list[str], name: str = "stub") -> None:
        self._replies = list(replies)
        self.name = name

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        if not self._replies:
            raise RuntimeError("No more replies")
        text = self._replies.pop(0)
        return Completion(
            text=text,
            model="stub-model",
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("stream() not implemented in test StubLLM")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.mark.asyncio
async def test_recording_llm_complete_records_and_delegates() -> None:
    """RecordingLLM.complete() delegates and records request+response."""
    stub = StubLLM(["reply1"], name="stub-provider")
    recorder = RecordingLLM(stub)

    messages = [Message(role=Role.USER, content="hello")]
    result = await recorder.complete(
        messages,
        model="test-model",
        system="You are helpful.",
        max_tokens=512,
        temperature=0.5,
    )

    # Delegation worked
    assert result.text == "reply1"
    assert result.model == "stub-model"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd == 0.001

    # Recording worked
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call.request.get("system") == "You are helpful."
    assert call.request.get("model") == "test-model"
    assert call.request.get("max_tokens") == 512
    assert call.request.get("temperature") == 0.5
    messages_obj = call.request.get("messages")
    assert isinstance(messages_obj, list)
    messages_list: list[object] = messages_obj
    assert len(messages_list) == 1
    msg_dict = messages_list[0]
    assert isinstance(msg_dict, dict)
    assert msg_dict.get("role") == "user"
    assert msg_dict.get("content") == "hello"

    assert call.response.get("text") == "reply1"
    assert call.response.get("model") == "stub-model"
    usage = call.response.get("usage")
    assert isinstance(usage, dict)
    assert usage.get("input_tokens") == 100
    assert usage.get("output_tokens") == 50
    assert usage.get("cost_usd") == 0.001


@pytest.mark.asyncio
async def test_recording_llm_stream_records_terminal_completion() -> None:
    """RecordingLLM.stream() delegates and records on terminal event."""

    class StreamingStub:
        def __init__(self) -> None:
            self.name = "streaming-stub"

        async def complete(
            self,
            messages: Sequence[Message],
            *,
            model: str | None = None,
            system: str | None = None,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> Completion:
            raise NotImplementedError()

        def stream(
            self,
            messages: Sequence[Message],
            *,
            model: str | None = None,
            system: str | None = None,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> AsyncIterator[StreamEvent]:
            return self._stream_impl()

        async def _stream_impl(self) -> AsyncIterator[StreamEvent]:
            # Yield delta
            yield StreamEvent(delta="hello")
            yield StreamEvent(delta=" world")
            # Terminal with completion
            completion = Completion(
                text="hello world",
                model="stream-model",
                usage=Usage(input_tokens=50, output_tokens=25, cost_usd=0.0005),
            )
            yield StreamEvent(completion=completion)

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    stub = StreamingStub()
    recorder = RecordingLLM(stub)

    messages = [Message(role=Role.USER, content="prompt")]
    events = []
    async for event in recorder.stream(messages, model="stream-model", max_tokens=256):
        events.append(event)

    # Should have recorded one call
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call.response.get("text") == "hello world"
    assert call.response.get("model") == "stream-model"


@pytest.mark.asyncio
async def test_playback_llm_complete_returns_fifo() -> None:
    """PlaybackLLM.complete() returns recorded completions in FIFO order."""
    calls = [
        RecordedCall(
            request={"messages": [], "model": None, "max_tokens": 1024, "temperature": 0.0},
            response={
                "text": "first",
                "model": "m1",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": None},
            },
        ),
        RecordedCall(
            request={"messages": [], "model": None, "max_tokens": 1024, "temperature": 0.0},
            response={
                "text": "second",
                "model": "m2",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10, "cost_usd": 0.002},
            },
        ),
    ]
    playback = PlaybackLLM(calls)

    c1 = await playback.complete([])
    assert c1.text == "first"
    assert c1.model == "m1"
    assert c1.usage.cost_usd is None

    c2 = await playback.complete([])
    assert c2.text == "second"
    assert c2.model == "m2"
    assert c2.usage.cost_usd == 0.002
    assert c2.stop_reason == "end_turn"

    # Third call should raise
    with pytest.raises(ReplayExhaustedError):
        await playback.complete([])


@pytest.mark.asyncio
async def test_playback_llm_exhausted_error_message() -> None:
    """ReplayExhaustedError carries message about exhaustion."""
    playback = PlaybackLLM([])

    with pytest.raises(ReplayExhaustedError) as exc_info:
        await playback.complete([])

    assert "exhausted" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_save_load_cassette_round_trip() -> None:
    """save_cassette/load_cassette round-trip preserves cost_usd=None."""
    calls = [
        RecordedCall(
            request={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "system": "You are helpful.",
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            response={
                "text": "Hi there",
                "model": "gpt-4",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50, "cost_usd": None},
            },
        ),
        RecordedCall(
            request={
                "messages": [{"role": "assistant", "content": "response"}],
                "model": None,
                "max_tokens": 2048,
                "temperature": 0.0,
            },
            response={
                "text": "Acknowledged",
                "model": "local-model",
                "stop_reason": None,
                "usage": {"input_tokens": 200, "output_tokens": 100, "cost_usd": 0.0},
            },
        ),
    ]

    tmp_path = Path("/tmp/test_cassette.jsonl")
    try:
        save_cassette(tmp_path, calls)
        loaded = load_cassette(tmp_path)

        assert len(loaded) == 2
        assert loaded[0].request.get("model") == "gpt-4"
        usage0 = loaded[0].response.get("usage")
        assert isinstance(usage0, dict)
        assert usage0.get("cost_usd") is None
        assert usage0.get("input_tokens") == 100
        usage1 = loaded[1].response.get("usage")
        assert isinstance(usage1, dict)
        assert usage1.get("cost_usd") == 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_playback_llm_stream_yields_delta_then_completion() -> None:
    """PlaybackLLM.stream() yields one delta event then terminal completion."""
    calls = [
        RecordedCall(
            request={"messages": [], "model": None, "max_tokens": 1024, "temperature": 0.0},
            response={
                "text": "streamed text",
                "model": "stream-model",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
            },
        ),
    ]
    playback = PlaybackLLM(calls)

    events = []
    async for event in playback.stream([]):
        events.append(event)

    assert len(events) == 2
    assert events[0].delta == "streamed text"
    assert events[0].completion is None
    assert events[1].completion is not None
    assert events[1].completion.text == "streamed text"
    assert events[1].completion.model == "stream-model"
    assert events[1].delta == ""


@pytest.mark.asyncio
async def test_recording_playback_identical_completions() -> None:
    """Record two completions, save, load, replay — completions are identical."""
    stub = StubLLM(
        ["answer1", "answer2"],
        name="original",
    )
    recorder = RecordingLLM(stub)

    # Record two calls
    msg1 = [Message(role=Role.USER, content="q1")]
    msg2 = [Message(role=Role.USER, content="q2")]
    c1 = await recorder.complete(msg1, model="m1", max_tokens=100)
    c2 = await recorder.complete(msg2, model="m2", max_tokens=200)

    # Save and load
    tmp_path = Path("/tmp/test_round_trip.jsonl")
    try:
        save_cassette(tmp_path, recorder.calls)
        loaded_calls = load_cassette(tmp_path)

        # Replay
        playback = PlaybackLLM(loaded_calls)
        p1 = await playback.complete([])
        p2 = await playback.complete([])

        # Completions should match
        assert p1.text == c1.text == "answer1"
        assert p1.model == c1.model == "stub-model"
        assert p1.usage.input_tokens == c1.usage.input_tokens == 100
        assert p1.usage.output_tokens == c1.usage.output_tokens == 50
        assert p1.usage.cost_usd == c1.usage.cost_usd == 0.001

        assert p2.text == c2.text == "answer2"
        assert p2.model == c2.model == "stub-model"
        assert p2.usage.input_tokens == c2.usage.input_tokens == 100
        assert p2.usage.output_tokens == c2.usage.output_tokens == 50
        assert p2.usage.cost_usd == c2.usage.cost_usd == 0.001
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_playback_from_completions_fixture() -> None:
    """PlaybackLLM.from_completions() allows hand-authored test fixtures."""
    completions = [
        Completion(
            text="fixture1",
            model="fixture-model",
            usage=Usage(input_tokens=1, output_tokens=2, cost_usd=None),
        ),
        Completion(
            text="fixture2",
            model="fixture-model",
            usage=Usage(input_tokens=3, output_tokens=4, cost_usd=0.0),
        ),
    ]

    playback = PlaybackLLM.from_completions(completions)
    c1 = await playback.complete([])
    c2 = await playback.complete([])

    assert c1.text == "fixture1"
    assert c1.usage.cost_usd is None
    assert c2.text == "fixture2"
    assert c2.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_playback_remaining_property() -> None:
    """PlaybackLLM.remaining tracks unplayed calls."""
    calls = [
        RecordedCall(
            request={"messages": [], "model": None, "max_tokens": 1024, "temperature": 0.0},
            response={
                "text": "1",
                "model": "m",
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": None},
            },
        ),
        RecordedCall(
            request={"messages": [], "model": None, "max_tokens": 1024, "temperature": 0.0},
            response={
                "text": "2",
                "model": "m",
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": None},
            },
        ),
    ]
    playback = PlaybackLLM(calls)

    assert playback.remaining == 2
    await playback.complete([])
    assert playback.remaining == 1
    await playback.complete([])
    assert playback.remaining == 0
