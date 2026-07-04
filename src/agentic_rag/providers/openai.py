"""OpenAI GPT provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import openai
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncStream,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from agentic_rag.config import Settings
from agentic_rag.providers.base import (
    Completion,
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderError,
    ProviderParseError,
    ProviderTimeoutError,
    Role,
    StreamEvent,
    Usage,
)
from agentic_rag.providers.pricing import cost_for
from agentic_rag.providers.retry import with_retries
from agentic_rag.tokens import count_tokens


def _to_chat_messages(
    messages: Sequence[Message], system: str | None
) -> list[ChatCompletionMessageParam]:
    api_messages: list[ChatCompletionMessageParam] = []
    if system:
        api_messages.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role is Role.USER:
            api_messages.append({"role": "user", "content": msg.content})
        else:
            api_messages.append({"role": "assistant", "content": msg.content})
    return api_messages


class OpenAIProvider:
    """GPT LLM provider via OpenAI API."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Lazy-initialize the async OpenAI client."""
        if self._client is None:
            self._client = openai.AsyncOpenAI()
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
        resolved_model = model or self.settings.openai.model
        client = self._get_client()

        @with_retries(self.settings.retry)
        async def _do_request() -> Completion:
            try:
                response = await client.chat.completions.create(
                    model=resolved_model,
                    messages=_to_chat_messages(messages, system),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                text = response.choices[0].message.content or ""
                stop_reason = response.choices[0].finish_reason

                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
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
        resolved_model = model or self.settings.openai.model
        client = self._get_client()

        @with_retries(self.settings.retry)
        async def _stream_connect() -> AsyncStream[ChatCompletionChunk]:
            try:
                stream = await client.chat.completions.create(
                    model=resolved_model,
                    messages=_to_chat_messages(messages, system),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                return stream
            except AuthenticationError as e:
                raise ProviderAuthError(str(e), provider=self.name) from e
            except (APITimeoutError, APIConnectionError) as e:
                raise ProviderTimeoutError(str(e), provider=self.name) from e
            except APIStatusError as e:
                raise ProviderAPIError(str(e), provider=self.name, status_code=e.status_code) from e
            except ProviderError:
                raise

        try:
            stream = await _stream_connect()

            text_accumulator = ""
            input_tokens = 0
            output_tokens = 0
            stop_reason = None

            async for chunk in stream:
                # Usage arrives on the final chunk when include_usage is set
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                # Process content delta
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        text_accumulator += choice.delta.content
                        yield StreamEvent(delta=choice.delta.content)

                    if choice.finish_reason:
                        stop_reason = choice.finish_reason

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

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderParseError(f"Stream error: {e}", provider=self.name) from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return count_tokens(text)
