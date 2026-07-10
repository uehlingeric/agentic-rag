"""Provider registry and factory functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from agentic_rag.config import Settings
from agentic_rag.providers.anthropic import AnthropicProvider
from agentic_rag.providers.base import (
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    ProviderParseError,
    ProviderTimeoutError,
    Usage,
)
from agentic_rag.providers.google import GoogleProvider
from agentic_rag.providers.ollama import OllamaProvider
from agentic_rag.providers.openai import OpenAIProvider
from agentic_rag.providers.pricing import cost_for
from agentic_rag.providers.stub import StubEmbeddingProvider, StubLLMProvider

if TYPE_CHECKING:
    import openai
    from google import genai

# Embedding model defaults by provider
_EMBEDDING_DEFAULTS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
    "google": "text-embedding-004",
}


def get_llm_provider(name: str, settings: Settings) -> LLMProvider:
    """Get an LLM provider by name.

    Args:
        name: Provider name: anthropic, openai, google, ollama, or stub.
        settings: Application settings.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    if name == "anthropic":
        return AnthropicProvider(settings)
    elif name == "openai":
        return OpenAIProvider(settings)
    elif name == "google":
        return GoogleProvider(settings)
    elif name == "ollama":
        return OllamaProvider(settings)
    elif name == "stub":
        return StubLLMProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: {name}. Valid options: anthropic, openai, google, ollama, stub"
        )


def get_embedding_provider(name: str, settings: Settings) -> EmbeddingProvider:
    """Get an embedding provider by name.

    Args:
        name: Provider name: openai, google, ollama, or stub.
        settings: Application settings.

    Returns:
        EmbeddingProvider instance.

    Raises:
        ValueError: If provider name is unknown or unsupported.
    """
    if name == "anthropic":
        raise ValueError(
            "Anthropic does not offer an embeddings API. "
            "Choose a different embedding provider (openai, google, ollama, stub)."
        )
    elif name == "openai":
        return OpenAIEmbeddingProvider(settings)
    elif name == "google":
        return GoogleEmbeddingProvider(settings)
    elif name == "ollama":
        return OllamaEmbeddingProvider(settings)
    elif name == "stub":
        return StubEmbeddingProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider: {name}. "
            f"Valid options: openai, google, ollama, stub (not anthropic)"
        )


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider."""

    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI()
        return self._client

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult:
        """Embed a batch of texts."""
        resolved_model = model or _EMBEDDING_DEFAULTS["openai"]

        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=resolved_model,
                input=list(texts),
            )

            vectors = [item.embedding for item in response.data]
            dimensions = len(vectors[0]) if vectors else 0

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            cost_usd = cost_for(self.name, resolved_model, input_tokens, 0)

            return EmbeddingResult(
                vectors=vectors,
                model=resolved_model,
                dimensions=dimensions,
                usage=Usage(input_tokens=input_tokens, output_tokens=0, cost_usd=cost_usd),
            )

        except Exception as e:
            raise ProviderParseError(f"OpenAI embedding error: {e}", provider=self.name) from e


class GoogleEmbeddingProvider:
    """Google Gemini embeddings provider."""

    name = "google"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Lazy-initialize Google client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult:
        """Embed a batch of texts."""
        resolved_model = model or _EMBEDDING_DEFAULTS["google"]

        try:
            client = self._get_client()

            response = await client.aio.models.embed_content(
                model=f"models/{resolved_model}",
                contents=list(texts),
            )

            vectors: list[list[float]] = []
            if response.embeddings:
                vectors = [item.values for item in response.embeddings if item.values is not None]
            dimensions = len(vectors[0]) if vectors else 0

            # Google doesn't provide token counts in embed_content, estimate from text
            total_chars = sum(len(t) for t in texts)
            estimated_tokens = total_chars // 4  # Rough estimate

            cost_usd = cost_for(self.name, resolved_model, estimated_tokens, 0)

            return EmbeddingResult(
                vectors=vectors,
                model=resolved_model,
                dimensions=dimensions,
                usage=Usage(
                    input_tokens=estimated_tokens,
                    output_tokens=0,
                    cost_usd=cost_usd,
                ),
            )

        except Exception as e:
            raise ProviderParseError(f"Google embedding error: {e}", provider=self.name) from e


class OllamaEmbeddingProvider:
    """Ollama local embeddings provider."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed_batch(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> EmbeddingResult:
        """Embed a batch of texts via Ollama /api/embed."""
        resolved_model = model or _EMBEDDING_DEFAULTS["ollama"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.settings.ollama.host}/api/embed",
                    json={
                        "model": resolved_model,
                        "input": list(texts),
                    },
                )

                if response.status_code != 200:
                    raise ProviderParseError(
                        f"Ollama returned {response.status_code}",
                        provider=self.name,
                    )

                data = response.json()
                embeddings = data.get("embeddings", [])
                vectors = embeddings if embeddings else []
                dimensions = len(vectors[0]) if vectors else 0

                return EmbeddingResult(
                    vectors=vectors,
                    model=resolved_model,
                    dimensions=dimensions,
                    usage=Usage(
                        input_tokens=0,  # Ollama doesn't track tokens for embeddings
                        output_tokens=0,
                        cost_usd=0.0,
                    ),
                )

        except (httpx.ConnectError, ConnectionError) as e:
            raise ProviderTimeoutError(
                f"Could not connect to Ollama at {self.settings.ollama.host}. "
                f"Is Ollama running? ({e})",
                provider=self.name,
            ) from e
        except ProviderParseError:
            raise
