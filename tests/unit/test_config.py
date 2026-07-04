from __future__ import annotations

from agentic_rag.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.provider == "ollama"
    assert s.chunking.target_tokens == 512
    assert s.chunking.overlap_tokens == 64


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RAG_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENTIC_RAG_OLLAMA__HOST", "http://gpu-box:11434")
    s = Settings()
    assert s.provider == "anthropic"
    assert s.ollama.host == "http://gpu-box:11434"


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
