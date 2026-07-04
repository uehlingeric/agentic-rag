"""Provider-agnostic contracts for LLM completion and embedding backends.

Every provider adapter implements these protocols. Downstream code (retrieval,
synthesis, agents, evals) depends only on this module — never on a vendor SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Role(StrEnum):
    """Conversation roles. System prompts are passed separately, not as messages."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for a single provider call.

    ``cost_usd`` is None when pricing is unknown; local providers report 0.0.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )

    @classmethod
    def zero(cls) -> Usage:
        return cls(input_tokens=0, output_tokens=0, cost_usd=0.0)


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    usage: Usage
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event from a streaming completion.

    Text events carry ``delta``; the final event carries the assembled
    ``completion`` (with usage) and an empty delta.
    """

    delta: str = ""
    completion: Completion | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[list[float]] = field(repr=False)
    model: str = ""
    dimensions: int = 0
    usage: Usage = field(default_factory=Usage.zero)


class ProviderError(Exception):
    """Base for all normalized provider failures.

    ``retryable`` drives the retry wrapper: rate limits, timeouts, and
    transient server errors retry; auth and validation failures do not.
    """

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ProviderAuthError(ProviderError):
    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=False)


class ProviderRateLimitError(ProviderError):
    def __init__(self, message: str, *, provider: str, retry_after: float | None = None) -> None:
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=True)


class ProviderAPIError(ProviderError):
    """Non-auth, non-rate-limit API failure. 5xx retryable, 4xx not."""

    def __init__(self, message: str, *, provider: str, status_code: int | None = None) -> None:
        retryable = status_code is not None and status_code >= 500
        super().__init__(message, provider=provider, retryable=retryable)
        self.status_code = status_code


class ProviderParseError(ProviderError):
    """Provider returned a payload we could not interpret."""

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=False)


@runtime_checkable
class LLMProvider(Protocol):
    """Chat-completion backend. ``model=None`` uses the configured default."""

    name: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]: ...

    def count_tokens(self, text: str) -> int:
        """Approximate token count (tiktoken locally; provider-neutral)."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Batch embedding backend. ``model=None`` uses the configured default."""

    name: str

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult: ...
