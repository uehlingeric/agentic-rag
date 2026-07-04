"""Tests for provider registry."""

import pytest

from agentic_rag.config import Settings
from agentic_rag.providers.anthropic import AnthropicProvider
from agentic_rag.providers.google import GoogleProvider
from agentic_rag.providers.ollama import OllamaProvider
from agentic_rag.providers.openai import OpenAIProvider
from agentic_rag.providers.registry import get_embedding_provider, get_llm_provider


class TestGetLLMProvider:
    """Test get_llm_provider factory."""

    def test_anthropic_provider(self) -> None:
        """Test getting Anthropic provider."""
        settings = Settings()
        provider = get_llm_provider("anthropic", settings)
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"

    def test_openai_provider(self) -> None:
        """Test getting OpenAI provider."""
        settings = Settings()
        provider = get_llm_provider("openai", settings)
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"

    def test_google_provider(self) -> None:
        """Test getting Google provider."""
        settings = Settings()
        provider = get_llm_provider("google", settings)
        assert isinstance(provider, GoogleProvider)
        assert provider.name == "google"

    def test_ollama_provider(self) -> None:
        """Test getting Ollama provider."""
        settings = Settings()
        provider = get_llm_provider("ollama", settings)
        assert isinstance(provider, OllamaProvider)
        assert provider.name == "ollama"

    def test_unknown_provider(self) -> None:
        """Test that unknown provider raises ValueError."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("unknown", settings)


class TestGetEmbeddingProvider:
    """Test get_embedding_provider factory."""

    def test_anthropic_not_supported(self) -> None:
        """Test that Anthropic embedding raises ValueError."""
        settings = Settings()
        with pytest.raises(ValueError, match="Anthropic does not offer"):
            get_embedding_provider("anthropic", settings)

    def test_openai_embedding_provider(self) -> None:
        """Test getting OpenAI embedding provider."""
        settings = Settings()
        provider = get_embedding_provider("openai", settings)
        assert provider.name == "openai"

    def test_google_embedding_provider(self) -> None:
        """Test getting Google embedding provider."""
        settings = Settings()
        provider = get_embedding_provider("google", settings)
        assert provider.name == "google"

    def test_ollama_embedding_provider(self) -> None:
        """Test getting Ollama embedding provider."""
        settings = Settings()
        provider = get_embedding_provider("ollama", settings)
        assert provider.name == "ollama"

    def test_unknown_embedding_provider(self) -> None:
        """Test that unknown embedding provider raises ValueError."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("unknown", settings)
