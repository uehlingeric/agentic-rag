"""OpenTelemetry instrumentation for agentic RAG pipelines.

Public API: tracer() and setup_tracing() for observability initialization.
"""

from __future__ import annotations

from agentic_rag.observability.tracing import set_usage_attributes, setup_tracing, tracer

__all__ = ["set_usage_attributes", "setup_tracing", "tracer"]
