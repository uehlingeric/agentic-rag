"""Render refusal messages from detections and policy templates.

Refusals are governed artifacts: downstream systems read Answer.refusal_reason
(machine key) while humans read the rendered template. Templates come from
policy so deployments can customize wording in guardrails.yaml without code
changes.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.guardrails.base import AppliedDetection, RefusalReason
from agentic_rag.guardrails.policy import GuardrailPolicy


def render_refusal(
    policy: GuardrailPolicy,
    reason: RefusalReason,
    applied: Sequence[AppliedDetection],
) -> str:
    """Render a refusal message from policy templates and detections.

    Templates use {entities} and {categories} str.format placeholders:
    - {entities}: comma-joined sorted unique PII entity types in applied
    - {categories}: comma-joined sorted unique injection categories in applied

    Args:
        policy: The active GuardrailPolicy.
        reason: The machine-readable RefusalReason.
        applied: AppliedDetections that triggered the refusal.

    Returns:
        Formatted refusal message ready to show the user.

    Raises:
        ValueError: If reason is OUT_OF_CORPUS (model's own refusal, no
            template defined).
    """
    if reason == RefusalReason.OUT_OF_CORPUS:
        raise ValueError("OUT_OF_CORPUS refusals are model-generated, not policy templates")

    # Extract unique entities and categories from applied detections
    entities = set()
    categories = set()

    for applied_det in applied:
        entity = applied_det.detection.entity
        if applied_det.detection.detector in ("regex", "ner"):
            # PII entity
            entities.add(entity)
        elif applied_det.detection.detector == "injection":
            # Injection category
            categories.add(entity)

    # Join sorted for deterministic output
    entities_str = ", ".join(sorted(entities))
    categories_str = ", ".join(sorted(categories))

    # Pick template by reason
    if reason == RefusalReason.INPUT_PII:
        template = policy.templates.input_pii
    elif reason == RefusalReason.INPUT_INJECTION:
        template = policy.templates.input_injection
    elif reason == RefusalReason.OUTPUT_PII:
        template = policy.templates.output_pii
    else:
        raise ValueError(f"Unknown refusal reason: {reason}")

    # Format template with entities and categories
    return template.format(entities=entities_str, categories=categories_str)
