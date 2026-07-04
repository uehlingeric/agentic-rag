"""Tests for embedding providers."""

import pytest
import respx
from httpx import Response

from agentic_rag.config import Settings
from agentic_rag.providers.registry import get_embedding_provider


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch) -> None:
    """Mock API keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


class TestOpenAIEmbedding:
    """Test OpenAI embedding provider."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self) -> None:
        """Test successful embedding batch."""
        settings = Settings()
        provider = get_embedding_provider("openai", settings)

        with respx.mock(base_url="https://api.openai.com") as respx_mock:
            respx_mock.post("/v1/embeddings").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                        ],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 20, "total_tokens": 20},
                    },
                )
            )

            result = await provider.embed_batch(["hello", "world"])

            assert len(result.vectors) == 2
            assert result.vectors[0] == [0.1, 0.2, 0.3]
            assert result.vectors[1] == [0.4, 0.5, 0.6]
            assert result.dimensions == 3
            assert result.usage.input_tokens == 20


class TestGoogleEmbedding:
    """Test Google embedding provider."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, monkeypatch) -> None:
        """Test successful embedding batch."""
        from unittest.mock import AsyncMock, MagicMock

        from google.genai import types as genai_types

        settings = Settings()
        provider = get_embedding_provider("google", settings)

        # Mock the client call, but use real response types to pin the SDK contract
        mock_response = genai_types.EmbedContentResponse(
            embeddings=[
                genai_types.ContentEmbedding(values=[0.1, 0.2, 0.3]),
                genai_types.ContentEmbedding(values=[0.4, 0.5, 0.6]),
            ]
        )

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

        result = await provider.embed_batch(["hello", "world"])

        assert len(result.vectors) == 2
        assert result.vectors[0] == [0.1, 0.2, 0.3]
        assert result.dimensions == 3


class TestOllamaEmbedding:
    """Test Ollama embedding provider."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self) -> None:
        """Test successful embedding batch."""
        settings = Settings()
        provider = get_embedding_provider("ollama", settings)

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/embed").mock(
                return_value=Response(
                    status_code=200,
                    json={
                        "embeddings": [
                            [0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6],
                        ]
                    },
                )
            )

            result = await provider.embed_batch(["hello", "world"])

            assert len(result.vectors) == 2
            assert result.vectors[0] == [0.1, 0.2, 0.3]
            assert result.dimensions == 3
            assert result.usage.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        """Test connection error."""
        from agentic_rag.providers.base import ProviderTimeoutError

        settings = Settings()
        provider = get_embedding_provider("ollama", settings)

        with respx.mock(base_url="http://localhost:11434") as respx_mock:
            respx_mock.post("/api/embed").mock(side_effect=ConnectionError())

            with pytest.raises(ProviderTimeoutError, match="Could not connect"):
                await provider.embed_batch(["hello", "world"])


class TestAnthropicEmbeddingNotSupported:
    """Test that Anthropic embedding is not available."""

    def test_anthropic_not_supported(self) -> None:
        """Test that Anthropic doesn't have embeddings."""
        from agentic_rag.providers.registry import get_embedding_provider

        settings = Settings()
        with pytest.raises(ValueError, match="Anthropic does not offer"):
            get_embedding_provider("anthropic", settings)
