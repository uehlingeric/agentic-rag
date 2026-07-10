"""Record and replay LLM provider calls for deterministic testing.

Use ``RecordingLLM`` to wrap a live provider and record all LLM calls to a
cassette file; then use ``PlaybackLLM`` to replay those calls deterministically
in tests, ensuring zero external dependencies and reproducible behavior.

Cassettes are JSONL-based (one call per line) and preserve all request/response
fields including ``cost_usd=None``.

Playback is FIFO order-based, not content-addressed; prompt edits invalidate
cassettes. This is intentional: it keeps the replay logic simple and makes
cassette divergence obvious (test failure = check if you changed the prompt).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.providers.base import Completion, LLMProvider, Message, StreamEvent, Usage
from agentic_rag.tokens import count_tokens


class ReplayExhaustedError(Exception):
    """Raised when PlaybackLLM runs out of recorded calls."""


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One recorded LLM call: request and response.

    Both request and response are stored as dicts for JSON serialization.
    Request keys: system, messages, model, max_tokens, temperature.
    Response keys: text, model, stop_reason, usage (with input_tokens,
    output_tokens, cost_usd).
    """

    request: dict[str, object]
    response: dict[str, object]


class RecordingLLM:
    """Wraps an LLMProvider to record all calls for later playback.

    Delegates to the inner provider and appends each call to a list that can
    be saved to disk via ``save_cassette()``.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self._calls: list[RecordedCall] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def calls(self) -> tuple[RecordedCall, ...]:
        """Immutable view of recorded calls."""
        return tuple(self._calls)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Delegate to inner provider and record the call."""
        completion = await self.inner.complete(
            messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._record_call(
            messages=messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            completion=completion,
        )
        return completion

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Delegate to inner provider stream and record the terminal completion."""
        return self._stream_wrapper(
            messages=messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _stream_wrapper(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream wrapper that records the terminal completion."""
        async for event in self.inner.stream(
            messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield event
            if event.completion is not None:
                self._record_call(
                    messages=messages,
                    model=model,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    completion=event.completion,
                )

    def count_tokens(self, text: str) -> int:
        """Delegate to inner provider."""
        return self.inner.count_tokens(text)

    def _record_call(
        self,
        messages: Sequence[Message],
        model: str | None,
        system: str | None,
        max_tokens: int,
        temperature: float,
        completion: Completion,
    ) -> None:
        """Append a RecordedCall to the list."""
        request_dict: dict[str, object] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            request_dict["system"] = system

        response_dict: dict[str, object] = {
            "text": completion.text,
            "model": completion.model,
            "stop_reason": completion.stop_reason,
            "usage": {
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "cost_usd": completion.usage.cost_usd,
            },
        }

        self._calls.append(RecordedCall(request=request_dict, response=response_dict))


class PlaybackLLM:
    """Plays back recorded LLM calls in FIFO order.

    Satisfies the LLMProvider protocol (duck-typed). Raises
    ``ReplayExhaustedError`` when no more calls are available.
    """

    def __init__(self, calls: Sequence[RecordedCall], name: str = "playback") -> None:
        self._calls = list(calls)
        self._name = name
        self._index = 0

    @classmethod
    def from_completions(cls, completions: Sequence[Completion]) -> PlaybackLLM:
        """Construct from hand-authored Completion objects.

        Convenience for fixture-based tests. Each Completion becomes a
        RecordedCall with a minimal request (model=None, system=None, etc.).
        """
        calls: list[RecordedCall] = []
        for completion in completions:
            request_dict: dict[str, object] = {
                "messages": [],
                "model": None,
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            response_dict: dict[str, object] = {
                "text": completion.text,
                "model": completion.model,
                "stop_reason": completion.stop_reason,
                "usage": {
                    "input_tokens": completion.usage.input_tokens,
                    "output_tokens": completion.usage.output_tokens,
                    "cost_usd": completion.usage.cost_usd,
                },
            }
            calls.append(RecordedCall(request=request_dict, response=response_dict))
        return cls(calls, name="playback-fixture")

    @property
    def name(self) -> str:
        return self._name

    @property
    def remaining(self) -> int:
        """Number of unplayed calls remaining."""
        return len(self._calls) - self._index

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Pop and return the next recorded completion."""
        if self._index >= len(self._calls):
            raise ReplayExhaustedError(
                f"No more calls available (exhausted after {len(self._calls)} calls)"
            )
        call = self._calls[self._index]
        self._index += 1
        return self._completion_from_response(call.response)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Pop and stream the next recorded completion.

        Yields one delta event with the full text, then the terminal event
        with the completion.
        """
        return self._stream_playback()

    async def _stream_playback(self) -> AsyncIterator[StreamEvent]:
        """Async generator for stream playback."""
        if self._index >= len(self._calls):
            raise ReplayExhaustedError(
                f"No more calls available (exhausted after {len(self._calls)} calls)"
            )
        call = self._calls[self._index]
        self._index += 1
        completion = self._completion_from_response(call.response)
        yield StreamEvent(delta=completion.text)
        yield StreamEvent(completion=completion)

    def count_tokens(self, text: str) -> int:
        """Use provider-neutral token counting."""
        return count_tokens(text)

    def _completion_from_response(self, response: dict[str, object]) -> Completion:
        """Reconstruct a Completion from a response dict."""
        usage_dict = response["usage"]
        if not isinstance(usage_dict, dict):
            raise ValueError(f"Invalid usage in response: {usage_dict}")
        usage = Usage(
            input_tokens=int(usage_dict["input_tokens"]),
            output_tokens=int(usage_dict["output_tokens"]),
            cost_usd=usage_dict.get("cost_usd"),
        )
        stop_reason = response.get("stop_reason")
        if stop_reason is not None:
            stop_reason = str(stop_reason)
        return Completion(
            text=str(response["text"]),
            model=str(response["model"]),
            usage=usage,
            stop_reason=stop_reason,
        )


def save_cassette(path: Path, calls: Sequence[RecordedCall]) -> None:
    """Write recorded calls to a JSONL cassette file.

    One call per line, fully JSON-serializable. Preserves cost_usd=None.

    Args:
        path: Output file path.
        calls: Sequence of RecordedCall objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for call in calls:
            line = json.dumps(
                {
                    "request": call.request,
                    "response": call.response,
                }
            )
            f.write(line + "\n")


def load_cassette(path: Path) -> tuple[RecordedCall, ...]:
    """Load recorded calls from a JSONL cassette file.

    Args:
        path: Input file path.

    Returns:
        Tuple of RecordedCall objects in order.

    Raises:
        FileNotFoundError: If the cassette file does not exist.
    """
    calls: list[RecordedCall] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            calls.append(
                RecordedCall(
                    request=data["request"],
                    response=data["response"],
                )
            )
    return tuple(calls)
