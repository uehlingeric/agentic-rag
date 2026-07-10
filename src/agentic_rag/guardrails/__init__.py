"""Guardrails: PII detection, prompt-injection screening, refusal policy,
and audit logging (week 6, ADR-008).

Safety sandwich: input scan (pre-planner) -> pipeline -> output scan
(post-critic), with every request writing an ``audit_v1`` record.
"""

from agentic_rag.guardrails.base import (
    Action,
    AppliedDetection,
    Detection,
    InjectionCategory,
    PIIEntity,
    RefusalReason,
    Verdict,
    redact,
)
from agentic_rag.guardrails.guarded import (
    GuardedPipeline,
    GuardedResult,
    GuardedStreamEvent,
)

__all__ = [
    "Action",
    "AppliedDetection",
    "Detection",
    "GuardedPipeline",
    "GuardedResult",
    "GuardedStreamEvent",
    "InjectionCategory",
    "PIIEntity",
    "RefusalReason",
    "Verdict",
    "redact",
]
