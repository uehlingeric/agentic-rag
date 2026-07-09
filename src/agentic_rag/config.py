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
    """``backend``: api | bedrock. Bedrock model IDs differ from API ones
    (e.g. ``us.anthropic.claude-...``), so ``bedrock_model`` must be set when
    ``backend`` is ``bedrock``. Auth uses the standard AWS credential chain."""

    model: str = "claude-sonnet-5"
    backend: str = "api"
    bedrock_model: str | None = None
    aws_region: str = "us-east-1"


class OpenAISettings(BaseModel):
    model: str = "gpt-5.4"


class GoogleSettings(BaseModel):
    """``backend``: api | vertex. Vertex authenticates via Application Default
    Credentials; ``vertex_project`` falls back to the ADC project when unset.
    Current Gemini models are served from the ``global`` endpoint, not regional
    ones (gemini-3.5-flash 404s in us-central1)."""

    model: str = "gemini-3.5-flash"
    backend: str = "api"
    vertex_project: str | None = None
    vertex_location: str = "global"


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


class JudgeSettings(BaseModel):
    """LLM-as-judge scoring (ADR-006). An answer is scored by the first entry in
    ``providers`` that differs from the provider that generated it, so headline
    numbers never come from a self-judging model."""

    providers: list[str] = ["anthropic", "google"]
    prompt_version: int | None = None
    max_tokens: int = 768
    max_parse_retries: int = 2


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
    judge: JudgeSettings = JudgeSettings()
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
