"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """POST /ask request body."""

    question: str = Field(..., min_length=1, max_length=4000)
    provider: str | None = None
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: Literal["none", "llm", "cross-encoder"] | None = None
    pipeline: Literal["vanilla", "agentic"] = "vanilla"
    stream: bool = False


class CitationOut(BaseModel):
    """Citation reference in the answer (same shape as the CLI --json record)."""

    marker: int
    chunk_id: str
    doc_id: str
    section_id: str
    heading: str
    page_start: int


class UsageOut(BaseModel):
    """Token usage and cost."""

    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class GuardrailScanOut(BaseModel):
    """Guardrail scan result for input or output."""

    detections: list[dict[str, object]]
    blocked: bool


class GuardrailsOut(BaseModel):
    """Full guardrails verdict."""

    input: GuardrailScanOut
    output: GuardrailScanOut | None
    retrieved_flagged_chunk_ids: list[str]


class AskResponse(BaseModel):
    """Successful /ask response."""

    question: str
    provider: str
    mode: str
    rerank: str
    pipeline: str
    answer: str
    refusal: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    invalid_citations: list[int]
    usage: UsageOut
    timings: dict[str, float]
    request_id: str
    guardrails: GuardrailsOut
    agent: dict[str, object] | None = None


class SearchResult(BaseModel):
    """Result from /search endpoint."""

    rank: int
    score: float
    chunk_id: str
    doc_id: str
    section_id: str
    heading: str
    page_start: int
    text: str


class StatsRow(BaseModel):
    """One aggregate row from the metrics ledger (MetricsStore.summary)."""

    group: str
    requests: int
    errors: int
    refusals: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_p50: float
    latency_p95: float


class StageRow(BaseModel):
    """Per-stage latency aggregate (MetricsStore.stage_summary)."""

    stage: str
    count: int
    p50: float
    p95: float


class StatsResponse(BaseModel):
    """GET /stats response."""

    summary: list[StatsRow]
    stages: list[StageRow] | None = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    index_loaded: bool
    provider: str
    version: str
