"""OpenTelemetry tracing setup and utilities.

Module-level flag prevents multiple setup() calls. setup_tracing() installs
a global TracerProvider with configured exporter and sampler; tracer() gets
the singleton trace.get_tracer("agentic_rag") for all instrumentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

import agentic_rag
from agentic_rag.providers.base import Usage

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

    from agentic_rag.config import Settings

# Module-level flag: set to True after setup_tracing() completes
_tracing_configured = False


def tracer() -> trace.Tracer:
    """Get the global tracer instance.

    Returns the singleton tracer for "agentic_rag". Instrumentation always
    compiles in; spans are no-ops when no provider is installed (disabled).

    Returns:
        Tracer for recording spans.
    """
    return trace.get_tracer("agentic_rag")


def setup_tracing(settings: Settings) -> bool:
    """Install the global OpenTelemetry TracerProvider.

    When observability is disabled or already configured, returns False (no-op).
    When enabled: builds a TracerProvider with the configured exporter (console
    or OTLP), sampler (ParentBased w/ head sampling), and service metadata;
    installs it globally; sets the module flag.

    Args:
        settings: Application settings with observability config.

    Returns:
        True if setup completed; False if disabled or already configured.

    Raises:
        ValueError: If exporter is not "console" or "otlp".
    """
    global _tracing_configured

    if not settings.observability.enabled:
        return False

    if _tracing_configured:
        return False

    provider = _build_provider(settings)
    trace.set_tracer_provider(provider)
    _tracing_configured = True

    return True


def _build_provider(settings: Settings) -> TracerProvider:
    """Build a TracerProvider with the configured exporter and sampler.

    Args:
        settings: Application settings with observability config.

    Returns:
        Configured TracerProvider ready for installation.

    Raises:
        ValueError: If exporter is not "console" or "otlp".
    """
    resource = Resource(
        attributes={
            "service.name": settings.observability.service_name,
            "service.version": agentic_rag.__version__,
        }
    )

    sampler = ParentBased(TraceIdRatioBased(settings.observability.sample_ratio))

    provider = TracerProvider(resource=resource, sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(settings)))

    return provider


def _build_exporter(settings: Settings) -> SpanExporter:
    """Build the configured span exporter.

    Args:
        settings: Application settings with observability config.

    Returns:
        ConsoleSpanExporter or OTLPSpanExporter (OTLP/HTTP).

    Raises:
        ValueError: If exporter is not "console" or "otlp".
    """
    if settings.observability.exporter == "console":
        import sys

        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        # stderr keeps stdout clean for answers and --json output
        return ConsoleSpanExporter(out=sys.stderr)
    if settings.observability.exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint + "/v1/traces")
    raise ValueError(f"Unknown exporter: {settings.observability.exporter}")


def set_usage_attributes(span: trace.Span, usage: Usage) -> None:
    """Set token and cost attributes on a span from a Usage dataclass.

    OTel attributes cannot be None, so ``rag.cost_usd`` is set only when
    ``cost_usd`` is known (local providers report None).

    Args:
        span: The span to annotate.
        usage: Usage with input_tokens, output_tokens, cost_usd.
    """
    span.set_attribute("rag.tokens.input", usage.input_tokens)
    span.set_attribute("rag.tokens.output", usage.output_tokens)

    if usage.cost_usd is not None:
        span.set_attribute("rag.cost_usd", usage.cost_usd)
