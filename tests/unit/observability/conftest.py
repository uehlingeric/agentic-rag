"""Shared tracing fixtures.

OpenTelemetry allows installing a global TracerProvider exactly once per
process (later ``set_tracer_provider`` calls are ignored with a warning), so
one provider with a single ``InMemorySpanExporter`` is installed lazily for
the whole test session; each test clears the exporter instead of trying to
reinstall a provider of its own.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_INSTALLED = False


@pytest.fixture(scope="session")
def _session_exporter() -> InMemorySpanExporter:
    """Install the process-wide test TracerProvider on first use."""
    global _INSTALLED
    if not _INSTALLED:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(provider)
        _INSTALLED = True
    return _EXPORTER


@pytest.fixture
def exporter(_session_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """The session exporter, cleared before and after each test."""
    _session_exporter.clear()
    yield _session_exporter
    _session_exporter.clear()
