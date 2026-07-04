"""Anthropic Claude provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncStream,
    AuthenticationError,
    RateLimitError,
    omit,
)
from anthropic.types import MessageParam, RawMessageStreamEvent

from agentic_rag.config import Settings
from agentic_rag.providers.base import (
    Completion,
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderError,
    ProviderTimeoutError,
    Role,
    StreamEvent,
    Usage,
)
from agentic_rag.providers.pricing import cost_for
from agentic_rag.providers.retry import with_retries
from agentic_rag.tokens import count_tokens


def _to_message_params(messages: Sequence[Message]) -> list[MessageParam]:
    return [
        {"role": "user" if msg.role is Role.USER else "assistant", "content": msg.content}
        for msg in messages
    ]


class AnthropicProvider:
    """Claude LLM provider via Anthropic API."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Lazy-initialize the async Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Generate a single completion."""
        resolved_model = model or self.settings.anthropic.model
        client = self._get_client()

        # Create the actual network call and wrap it with retries
        @with_retries(self.settings.retry)
        async def _do_request() -> Completion:
            try:
                response = await client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=_to_message_params(messages),
                    temperature=temperature,
                    system=system if system is not None else omit,
                )

                text = "".join(block.text for block in response.content if block.type == "text")
                stop_reason = response.stop_reason

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost_usd = cost_for(self.name, resolved_model, input_tokens, output_tokens)

                return Completion(
                    text=text,
                    model=resolved_model,
                    usage=Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
                    ),
                    stop_reason=stop_reason,
                )

            except AuthenticationError as e:
                raise ProviderAuthError(str(e), provider=self.name) from e
            except RateLimitError as e:
                raise ProviderError(str(e), provider=self.name, retryable=True) from e
            except (APITimeoutError, APIConnectionError) as e:
                raise ProviderTimeoutError(str(e), provider=self.name) from e
            except APIStatusError as e:
                raise ProviderAPIError(str(e), provider=self.name, status_code=e.status_code) from e
            except ProviderError:
                raise

        return await _do_request()

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion as events."""
        resolved_model = model or self.settings.anthropic.model
        client = self._get_client()

        # Only the initial connection is retried; a stream that fails mid-flight
        # after yielding text cannot be transparently restarted.
        @with_retries(self.settings.retry)
        async def _stream_connect() -> AsyncStream[RawMessageStreamEvent]:
            try:
                return await client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=_to_message_params(messages),
                    temperature=temperature,
                    system=system if system is not None else omit,
                    stream=True,
                )
            except AuthenticationError as e:
                raise ProviderAuthError(str(e), provider=self.name) from e
            except RateLimitError as e:
                raise ProviderError(str(e), provider=self.name, retryable=True) from e
            except (APITimeoutError, APIConnectionError) as e:
                raise ProviderTimeoutError(str(e), provider=self.name) from e
            except APIStatusError as e:
                raise ProviderAPIError(str(e), provider=self.name, status_code=e.status_code) from e

        stream = await _stream_connect()

        text_accumulator = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        try:
            async for event in stream:
                if event.type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                    text_accumulator += event.delta.text
                    yield StreamEvent(delta=event.delta.text)
                elif event.type == "message_delta":
                    output_tokens = event.usage.output_tokens
                    if event.delta.stop_reason is not None:
                        stop_reason = event.delta.stop_reason
        except (APITimeoutError, APIConnectionError) as e:
            raise ProviderTimeoutError(str(e), provider=self.name) from e
        except APIStatusError as e:
            raise ProviderAPIError(str(e), provider=self.name, status_code=e.status_code) from e

        cost_usd = cost_for(self.name, resolved_model, input_tokens, output_tokens)

        yield StreamEvent(
            delta="",
            completion=Completion(
                text=text_accumulator,
                model=resolved_model,
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                ),
                stop_reason=stop_reason,
            ),
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return count_tokens(text)
