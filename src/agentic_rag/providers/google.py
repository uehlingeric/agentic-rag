"""Google Gemini provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import google.genai
from google.genai import errors as genai_errors
from google.genai import types

from agentic_rag.config import Settings
from agentic_rag.providers.base import (
    Completion,
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderError,
    ProviderParseError,
    ProviderTimeoutError,
    StreamEvent,
    Usage,
)
from agentic_rag.providers.pricing import cost_for
from agentic_rag.providers.retry import with_retries
from agentic_rag.tokens import count_tokens


class GoogleProvider:
    """Gemini LLM provider via the Google AI API or Vertex AI."""

    name = "google"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: google.genai.Client | None = None

    def _get_client(self) -> google.genai.Client:
        """Lazy-initialize the client for the configured backend."""
        if self._client is None:
            cfg = self.settings.google
            if cfg.backend == "api":
                self._client = google.genai.Client()
            elif cfg.backend == "vertex":
                self._client = google.genai.Client(
                    vertexai=True,
                    project=cfg.vertex_project,
                    location=cfg.vertex_location,
                )
            else:
                raise ValueError(
                    f"Unknown google backend: {cfg.backend}. Valid options: api, vertex"
                )
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
        resolved_model = model or self.settings.google.model
        client = self._get_client()

        @with_retries(self.settings.retry)
        async def _do_request() -> Completion:
            try:
                # Convert messages to Google format
                api_messages = [
                    {"role": msg.role.value, "parts": [{"text": msg.content}]} for msg in messages
                ]

                response = await client.aio.models.generate_content(
                    model=resolved_model,
                    contents=api_messages,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )

                text = response.text or ""
                stop_reason = None
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason is not None:
                        stop_reason = finish_reason.name

                # Extract usage from usage_metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0

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

            except Exception as e:
                # Handle timeouts first before other exception types
                if "timeout" in str(e).lower() or "deadline exceeded" in str(e).lower():
                    raise ProviderTimeoutError(str(e), provider=self.name) from e
                # Then handle API errors
                if isinstance(e, genai_errors.APIError):
                    if hasattr(e, "code"):
                        if e.code == 401 or "authentication" in str(e).lower():
                            raise ProviderAuthError(str(e), provider=self.name) from e
                        elif e.code == 429:
                            raise ProviderError(str(e), provider=self.name, retryable=True) from e
                        elif e.code >= 500:
                            raise ProviderAPIError(
                                str(e), provider=self.name, status_code=e.code
                            ) from e
                    raise ProviderAPIError(str(e), provider=self.name, status_code=None) from e
                # If it's already a provider error, re-raise
                if isinstance(e, ProviderError):
                    raise
                # Should not reach here
                raise ProviderParseError(f"Unexpected error: {e}", provider=self.name) from e

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
        resolved_model = model or self.settings.google.model
        client = self._get_client()

        @with_retries(self.settings.retry)
        async def _stream_connect() -> AsyncIterator[types.GenerateContentResponse]:
            api_messages = [
                {"role": msg.role.value, "parts": [{"text": msg.content}]} for msg in messages
            ]

            try:
                stream = await client.aio.models.generate_content_stream(
                    model=resolved_model,
                    contents=api_messages,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return stream
            except Exception as e:
                # Handle timeouts first
                if "timeout" in str(e).lower() or "deadline exceeded" in str(e).lower():
                    raise ProviderTimeoutError(str(e), provider=self.name) from e
                # Handle API errors
                if isinstance(e, genai_errors.APIError):
                    if hasattr(e, "code") and e.code == 401:
                        raise ProviderAuthError(str(e), provider=self.name) from e
                    raise ProviderAPIError(str(e), provider=self.name, status_code=None) from e
                # Re-raise provider errors
                if isinstance(e, ProviderError):
                    raise
                # Should not reach here
                raise ProviderParseError(f"Stream error: {e}", provider=self.name) from e

        try:
            stream = await _stream_connect()

            text_accumulator = ""
            input_tokens = 0
            output_tokens = 0
            stop_reason = None

            async for chunk in stream:
                # Process text parts
                if chunk.text:
                    text_accumulator += chunk.text
                    yield StreamEvent(delta=chunk.text)

                # Get final usage info
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    input_tokens = chunk.usage_metadata.prompt_token_count or 0
                    output_tokens = chunk.usage_metadata.candidates_token_count or 0

                # Get finish reason from last chunk
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "finish_reason") and candidate.finish_reason is not None:
                        stop_reason = candidate.finish_reason.name

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

        except Exception as e:
            # Handle timeouts first
            if "timeout" in str(e).lower() or "deadline exceeded" in str(e).lower():
                raise ProviderTimeoutError(str(e), provider=self.name) from e
            if isinstance(e, genai_errors.APIError):
                raise ProviderAPIError(str(e), provider=self.name, status_code=None) from e
            # ProviderError and anything unrecognized propagate unchanged
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return count_tokens(text)
