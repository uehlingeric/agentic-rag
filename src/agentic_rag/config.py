"""Application settings: code defaults < config.yaml < environment variables.

API keys are never part of this config — vendor SDKs read them from standard
environment variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``GOOGLE_API_KEY``). A local ``.env`` is loaded into the process environment
for convenience; it is gitignored.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class AnthropicSettings(BaseModel):
    model: str = "claude-sonnet-5"


class OpenAISettings(BaseModel):
    model: str = "gpt-5.4"


class GoogleSettings(BaseModel):
    model: str = "gemini-3.5-flash"


class OllamaSettings(BaseModel):
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"


class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"


class ChunkingSettings(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64


class RetrievalSettings(BaseModel):
    """``candidate_pool`` is the per-mode depth fed into RRF before the final cut."""

    top_k: int = 10
    candidate_pool: int = 50
    rrf_k: int = 60


class RerankSettings(BaseModel):
    """``mode``: none | llm | cross-encoder. ``model`` overrides the backend default."""

    mode: str = "none"
    candidate_pool: int = 30
    top_k: int = 8
    model: str | None = None


class SynthesisSettings(BaseModel):
    max_context_tokens: int = 6000
    max_answer_tokens: int = 1024


class RetrySettings(BaseModel):
    max_attempts: int = 5
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        extra="ignore",
    )

    provider: str = "ollama"
    anthropic: AnthropicSettings = AnthropicSettings()
    openai: OpenAISettings = OpenAISettings()
    google: GoogleSettings = GoogleSettings()
    ollama: OllamaSettings = OllamaSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    rerank: RerankSettings = RerankSettings()
    synthesis: SynthesisSettings = SynthesisSettings()
    retry: RetrySettings = RetrySettings()
    data_dir: Path = Path("data")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings()
