"""Tests for LLM adapter implementations."""

import json

import pytest
import respx
from httpx import Response

from agentic_rag.config import RetrySettings, Settings
from agentic_rag.providers.anthropic import AnthropicProvider
from agentic_rag.providers.base import (
    Message,
    ProviderAPIError,
    ProviderAuthError,
    ProviderTimeoutError,
    Role,
)
from agentic_rag.providers.google import GoogleProvider
from agentic_rag.providers.ollama import OllamaProvider
from agentic_rag.providers.openai import OpenAIProvider


# Fixtures
@pytest.fixture
def settings_no_retry() -> Settings:
    """Settings with no retries for fast tests."""
    return Settings(retry=RetrySettings(max_attempts=1, initial_backoff_s=0.0, max_backoff_s=0.0))


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch) -> None:
    """Mock API keys for all providers."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-google")


# Anthropic tests
class TestAnthropicProvider:
    """Test Anthropic Claude provider."""

    @pytest.mark.asyncio
    async def test_complete_success(self, settings_no_retry: Settings) -> None:
        """Test successful completion."""
        provider = AnthropicProvider(settings_no_retry)

        with respx.mock(base_url="https://api.anthropic.com") as respx_mock:
            respx_mock.post("/v1/messages").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello world"}],
                        "model": "claude-sonnet-5",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )
            )

            completion = await provider.complete(
                [Message(role=Role.USER, content="Hello")],
                model="claude-sonnet-5",
            )

            assert completion.text == "Hello world"
            assert completion.model == "claude-sonnet-5"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5
            assert completion.usage.cost_usd is not None

    @pytest.mark.asyncio
    async def test_complete_auth_error(self, settings_no_retry: Settings) -> None:
        """Test authentication error."""
        provider = AnthropicProvider(settings_no_retry)

        with respx.mock(base_url="https://api.anthropic.com") as respx_mock:
            respx_mock.post("/v1/messages").mock(return_value=Response(status_code=401))

            with pytest.raises(ProviderAuthError):
                await provider.complete([Message(role=Role.USER, content="test")])

    @pytest.mark.asyncio
    async def test_complete_timeout(self, settings_no_retry: Settings) -> None:
        """Test timeout error."""
        provider = AnthropicProvider(settings_no_retry)

        with respx.mock(base_url="https://api.anthropic.com") as respx_mock:
            respx_mock.post("/v1/messages").mock(side_effect=TimeoutError())

            with pytest.raises(ProviderTimeoutError):
                await provider.complete([Message(role=Role.USER, content="test")])

    @pytest.mark.asyncio
    async def test_complete_with_system(self, settings_no_retry: Settings) -> None:
        """Test completion with system prompt."""
        provider = AnthropicProvider(settings_no_retry)

        with respx.mock(base_url="https://api.anthropic.com") as respx_mock:
            respx_mock.post("/v1/messages").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "id": "msg_1",
                        "content": [{"type": "text", "text": "Response"}],
                        "model": "claude-sonnet-5",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )
            )

            completion = await provider.complete(
                [Message(role=Role.USER, content="test")],
                system="You are helpful",
            )

            assert completion.text == "Response"

    @pytest.mark.asyncio
    async def test_stream_success(self, settings_no_retry: Settings) -> None:
        """Test streaming via the real SSE wire format."""
        provider = AnthropicProvider(settings_no_retry)

        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        sse_body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)

        with respx.mock(base_url="https://api.anthropic.com") as respx_mock:
            respx_mock.post("/v1/messages").mock(
                return_value=Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    text=sse_body,
                )
            )

            deltas = []
            completion = None
            async for event in provider.stream(
                [Message(role=Role.USER, content="Hello")],
                model="claude-sonnet-5",
            ):
                if event.delta:
                    deltas.append(event.delta)
                if event.completion:
                    completion = event.completion

            assert deltas == ["Hello", " world"]
            assert completion is not None
            assert completion.text == "Hello world"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5
            assert completion.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_count_tokens(self, settings_no_retry: Settings) -> None:
        """Test token counting."""
        provider = AnthropicProvider(settings_no_retry)
        count = provider.count_tokens("hello world")
        assert count > 0


# OpenAI tests
class TestOpenAIProvider:
    """Test OpenAI GPT provider."""

    @pytest.mark.asyncio
    async def test_complete_success(self, settings_no_retry: Settings) -> None:
        """Test successful completion."""
        provider = OpenAIProvider(settings_no_retry)

        with respx.mock(base_url="https://api.openai.com") as respx_mock:
            respx_mock.post("/v1/chat/completions").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 123456,
                        "model": "gpt-5.4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Hello!"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )
            )

            completion = await provider.complete(
                [Message(role=Role.USER, content="Hello")],
                model="gpt-5.4",
            )

            assert completion.text == "Hello!"
            assert completion.model == "gpt-5.4"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_auth_error(self, settings_no_retry: Settings) -> None:
        """Test authentication error."""
        provider = OpenAIProvider(settings_no_retry)

        with respx.mock(base_url="https://api.openai.com") as respx_mock:
            respx_mock.post("/v1/chat/completions").mock(return_value=Response(status_code=401))

            with pytest.raises(ProviderAuthError):
                await provider.complete([Message(role=Role.USER, content="test")])

    @pytest.mark.asyncio
    async def test_stream_success(self, settings_no_retry: Settings) -> None:
        """Test streaming via the real SSE wire format."""
        provider = OpenAIProvider(settings_no_retry)

        def _chunk(delta: dict, finish_reason: str | None) -> dict:
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 123456,
                "model": "gpt-5.4",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        usage_chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123456,
            "model": "gpt-5.4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        payloads = [
            _chunk({"role": "assistant", "content": ""}, None),
            _chunk({"content": "Hello"}, None),
            _chunk({"content": " world"}, None),
            _chunk({}, "stop"),
            usage_chunk,
        ]
        sse_body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads) + "data: [DONE]\n\n"

        with respx.mock(base_url="https://api.openai.com") as respx_mock:
            respx_mock.post("/v1/chat/completions").mock(
                return_value=Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    text=sse_body,
                )
            )

            deltas = []
            completion = None
            async for event in provider.stream(
                [Message(role=Role.USER, content="Hello")],
                model="gpt-5.4",
            ):
                if event.delta:
                    deltas.append(event.delta)
                if event.completion:
                    completion = event.completion

            assert deltas == ["Hello", " world"]
            assert completion is not None
            assert completion.text == "Hello world"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5
            assert completion.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_count_tokens(self, settings_no_retry: Settings) -> None:
        """Test token counting."""
        provider = OpenAIProvider(settings_no_retry)
        count = provider.count_tokens("hello world")
        assert count > 0


# Google tests
class TestGoogleProvider:
    """Test Google Gemini provider."""

    @pytest.mark.asyncio
    async def test_complete_success(self, settings_no_retry: Settings) -> None:
        """Test successful completion."""
        provider = GoogleProvider(settings_no_retry)

        with respx.mock(base_url="https://generativelanguage.googleapis.com") as respx_mock:
            respx_mock.post("/v1beta/models/gemini-3-5-flash:generateContent").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "Hello world"}], "role": "model"},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 5,
                        },
                    },
                )
            )

            completion = await provider.complete(
                [Message(role=Role.USER, content="Hello")],
                model="gemini-3-5-flash",
            )

            assert completion.text == "Hello world"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_count_tokens(self, settings_no_retry: Settings) -> None:
        """Test token counting."""
        provider = GoogleProvider(settings_no_retry)
        count = provider.count_tokens("hello world")
        assert count > 0


# Ollama tests
class TestOllamaProvider:
    """Test Ollama local provider."""

    @pytest.mark.asyncio
    async def test_complete_success(self, settings_no_retry: Settings) -> None:
        """Test successful completion."""
        provider = OllamaProvider(settings_no_retry)

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/chat").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "model": "llama3.1:8b",
                        "message": {"role": "assistant", "content": "Hello!"},
                        "done": True,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                )
            )

            completion = await provider.complete(
                [Message(role=Role.USER, content="Hello")],
                model="llama3.1:8b",
            )

            assert completion.text == "Hello!"
            assert completion.model == "llama3.1:8b"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5
            assert completion.usage.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_complete_connection_error(self, settings_no_retry: Settings) -> None:
        """Test connection error."""
        provider = OllamaProvider(settings_no_retry)

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/chat").mock(side_effect=ConnectionError())

            with pytest.raises(ProviderTimeoutError, match="Could not connect"):
                await provider.complete([Message(role=Role.USER, content="test")])

    @pytest.mark.asyncio
    async def test_stream_success(self, settings_no_retry: Settings) -> None:
        """Test successful streaming."""
        provider = OllamaProvider(settings_no_retry)

        # Stream response as NDJSON
        stream_data = "\n".join(
            [
                json.dumps(
                    {
                        "model": "llama3.1:8b",
                        "message": {"role": "assistant", "content": "Hello"},
                        "done": False,
                    }
                ),
                json.dumps(
                    {
                        "model": "llama3.1:8b",
                        "message": {"role": "assistant", "content": " world"},
                        "done": True,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    }
                ),
            ]
        )

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/chat").mock(
                return_value=Response(status_code=200, text=stream_data)
            )

            deltas = []
            completion = None
            async for event in provider.stream(
                [Message(role=Role.USER, content="Hello")],
                model="llama3.1:8b",
            ):
                if event.delta:
                    deltas.append(event.delta)
                if event.completion:
                    completion = event.completion

            assert deltas == ["Hello", " world"]
            assert completion is not None
            assert completion.text == "Hello world"
            assert completion.usage.input_tokens == 10
            assert completion.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_count_tokens(self, settings_no_retry: Settings) -> None:
        """Test token counting."""
        provider = OllamaProvider(settings_no_retry)
        count = provider.count_tokens("hello world")
        assert count > 0

    @pytest.mark.asyncio
    async def test_complete_http_error(self, settings_no_retry: Settings) -> None:
        """Test HTTP error handling."""
        provider = OllamaProvider(settings_no_retry)

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/chat").mock(return_value=Response(status_code=500))

            with pytest.raises(ProviderAPIError):
                await provider.complete([Message(role=Role.USER, content="test")])
