"""Tests for tracing setup: provider/exporter construction and usage attributes."""

from __future__ import annotations

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agentic_rag.observability.tracing as tracing_module
from agentic_rag.config import ObservabilitySettings, Settings
from agentic_rag.observability import set_usage_attributes, setup_tracing, tracer
from agentic_rag.observability.tracing import _build_exporter, _build_provider
from agentic_rag.providers.base import Usage


def obs_settings(**kwargs: object) -> Settings:
    return Settings(observability=ObservabilitySettings(**kwargs))  # type: ignore[arg-type]


class TestSetupTracing:
    def test_disabled_returns_false(self) -> None:
        assert setup_tracing(obs_settings(enabled=False)) is False

    def test_enabled_installs_provider_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First enabled call installs a provider; the second is a no-op.

        ``set_tracer_provider`` is monkeypatched so the test never touches the
        real process-wide provider (which can only be installed once).
        """
        installed: list[object] = []
        monkeypatch.setattr(tracing_module.trace, "set_tracer_provider", installed.append)
        monkeypatch.setattr(tracing_module, "_tracing_configured", False)

        assert setup_tracing(obs_settings(enabled=True)) is True
        assert len(installed) == 1
        assert isinstance(installed[0], TracerProvider)

        assert setup_tracing(obs_settings(enabled=True)) is False
        assert len(installed) == 1


class TestBuildProvider:
    def test_resource_carries_service_name_and_version(self) -> None:
        provider = _build_provider(obs_settings(service_name="test-svc"))
        assert provider.resource.attributes["service.name"] == "test-svc"
        assert "service.version" in provider.resource.attributes

    def test_sampler_uses_configured_ratio(self) -> None:
        provider = _build_provider(obs_settings(sample_ratio=0.25))
        assert "0.25" in provider.sampler.get_description()

    def test_console_exporter(self) -> None:
        assert isinstance(_build_exporter(obs_settings(exporter="console")), ConsoleSpanExporter)

    def test_otlp_exporter(self) -> None:
        exporter = _build_exporter(
            obs_settings(exporter="otlp", otlp_endpoint="http://jaeger:4318")
        )
        assert isinstance(exporter, OTLPSpanExporter)

    def test_unknown_exporter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown exporter"):
            _build_exporter(obs_settings(exporter="bogus"))


class TestSetUsageAttributes:
    def test_sets_tokens_and_cost(self, exporter: InMemorySpanExporter) -> None:
        with tracer().start_as_current_span("test") as span:
            set_usage_attributes(span, Usage(input_tokens=100, output_tokens=50, cost_usd=1.23))
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes["rag.tokens.input"] == 100
        assert finished.attributes["rag.tokens.output"] == 50
        assert finished.attributes["rag.cost_usd"] == 1.23

    def test_omits_cost_when_none(self, exporter: InMemorySpanExporter) -> None:
        with tracer().start_as_current_span("test") as span:
            set_usage_attributes(span, Usage(input_tokens=10, output_tokens=5, cost_usd=None))
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes["rag.tokens.input"] == 10
        assert "rag.cost_usd" not in finished.attributes
