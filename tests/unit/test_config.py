from __future__ import annotations

from agentic_rag.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.provider == "ollama"
    assert s.chunking.target_tokens == 512
    assert s.chunking.overlap_tokens == 64


def test_backend_defaults() -> None:
    s = Settings()
    assert s.anthropic.backend == "api"
    assert s.anthropic.bedrock_model is None
    assert s.anthropic.aws_region == "us-east-1"
    assert s.google.backend == "api"
    assert s.google.vertex_project is None
    assert s.google.vertex_location == "global"


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENTIC_RAG_OLLAMA__HOST", "http://gpu-box:11434")
    monkeypatch.setenv("AGENTIC_RAG_ANTHROPIC__BACKEND", "bedrock")
    monkeypatch.setenv("AGENTIC_RAG_GOOGLE__BACKEND", "vertex")
    s = Settings()
    assert s.provider == "anthropic"
    assert s.ollama.host == "http://gpu-box:11434"
    assert s.anthropic.backend == "bedrock"
    assert s.google.backend == "vertex"


def test_usage_addition() -> None:
    from agentic_rag.providers import Usage

    a = Usage(input_tokens=10, output_tokens=5, cost_usd=0.01)
    b = Usage(input_tokens=1, output_tokens=2)
    total = a + b
    assert total.input_tokens == 11
    assert total.output_tokens == 7
    assert total.cost_usd == 0.01
    assert (
        Usage(input_tokens=1, output_tokens=1) + Usage(input_tokens=1, output_tokens=1)
    ).cost_usd is None


def test_observability_defaults() -> None:
    from agentic_rag.config import ObservabilitySettings

    obs = ObservabilitySettings()
    assert obs.enabled is False
    assert obs.exporter == "console"
    assert obs.otlp_endpoint == "http://localhost:4318"
    assert obs.sample_ratio == 1.0
    assert obs.service_name == "agentic-rag"


def test_observability_in_settings() -> None:
    s = Settings()
    assert s.observability is not None
    assert s.observability.enabled is False
    assert s.observability.exporter == "console"
