"""Ollama local LLM provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from agentic_rag.config import Settings
from agentic_rag.providers.base import (
    Completion,
    Message,
    ProviderAPIError,
    ProviderError,
    ProviderParseError,
    ProviderTimeoutError,
    StreamEvent,
    Usage,
)
from agentic_rag.providers.retry import with_retries
from agentic_rag.tokens import count_tokens


class OllamaProvider:
    """Ollama local LLM provider using raw httpx."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        resolved_model = model or self.settings.ollama.model

        @with_retries(self.settings.retry)
        async def _do_request() -> Completion:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Convert messages to Ollama format
                    api_messages = []

                    if system:
                        api_messages.append({"role": "system", "content": system})

                    for msg in messages:
                        api_messages.append({"role": msg.role.value, "content": msg.content})

                    payload = {
                        "model": resolved_model,
                        "messages": api_messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    }

                    response = await client.post(
                        f"{self.settings.ollama.host}/api/chat",
                        json=payload,
                    )

                    if response.status_code != 200:
                        raise ProviderAPIError(
                            f"Ollama returned {response.status_code}",
                            provider=self.name,
                            status_code=response.status_code,
                        )

                    data = response.json()
                    text = data.get("message", {}).get("content", "")

                    # Get token counts from the response
                    input_tokens = data.get("prompt_eval_count", 0)
                    output_tokens = data.get("eval_count", 0)

                    return Completion(
                        text=text,
                        model=resolved_model,
                        usage=Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost_usd=0.0,  # Ollama is local
                        ),
                        stop_reason=data.get("done_reason"),
                    )

            except (httpx.ConnectError, ConnectionError) as e:
                raise ProviderTimeoutError(
                    f"Could not connect to Ollama at {self.settings.ollama.host}. "
                    f"Is Ollama running? ({e})",
                    provider=self.name,
                ) from e
            except (httpx.TimeoutException, TimeoutError) as e:
                raise ProviderTimeoutError(str(e), provider=self.name) from e
            except httpx.HTTPError as e:
                raise ProviderAPIError(str(e), provider=self.name) from e
            except (json.JSONDecodeError, KeyError) as e:
                raise ProviderParseError(
                    f"Failed to parse Ollama response: {e}", provider=self.name
                ) from e
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
        """Stream a completion as NDJSON events."""
        resolved_model = model or self.settings.ollama.model

        # No retry wrapper here: the connection is only established inside
        # client.stream(), and a stream that has already yielded text cannot
        # be transparently restarted. Connect errors are normalized below.
        client = httpx.AsyncClient(timeout=30.0)

        try:
            api_messages = []

            if system:
                api_messages.append({"role": "system", "content": system})

            for msg in messages:
                api_messages.append({"role": msg.role.value, "content": msg.content})

            payload = {
                "model": resolved_model,
                "messages": api_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            text_accumulator = ""
            input_tokens = 0
            output_tokens = 0
            stop_reason: str | None = None

            async with client.stream(
                "POST",
                f"{self.settings.ollama.host}/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise ProviderAPIError(
                        f"Ollama returned {response.status_code}",
                        provider=self.name,
                        status_code=response.status_code,
                    )

                # Parse NDJSON stream
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        message = event.get("message", {})
                        content = message.get("content", "")

                        if content:
                            text_accumulator += content
                            yield StreamEvent(delta=content)

                        # Capture final token counts and stop reason
                        if event.get("done", False):
                            input_tokens = event.get("prompt_eval_count", 0)
                            output_tokens = event.get("eval_count", 0)
                            stop_reason = event.get("done_reason")

                    except json.JSONDecodeError as e:
                        raise ProviderParseError(
                            f"Failed to parse Ollama NDJSON: {e}", provider=self.name
                        ) from e

            # Final event with completion
            yield StreamEvent(
                delta="",
                completion=Completion(
                    text=text_accumulator,
                    model=resolved_model,
                    usage=Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=0.0,
                    ),
                    stop_reason=stop_reason,
                ),
            )

        except ProviderTimeoutError:
            raise
        except (httpx.ConnectError, ConnectionError) as e:
            raise ProviderTimeoutError(
                f"Could not connect to Ollama at {self.settings.ollama.host}. "
                f"Is Ollama running? ({e})",
                provider=self.name,
            ) from e
        except ProviderParseError:
            raise
        except ProviderError:
            raise
        finally:
            await client.aclose()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return count_tokens(text)
