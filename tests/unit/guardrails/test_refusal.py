"""Tests for rendering refusal messages from policy templates."""

from __future__ import annotations

import pytest

from agentic_rag.guardrails.base import (
    Action,
    AppliedDetection,
    Detection,
    RefusalReason,
)
from agentic_rag.guardrails.policy import GuardrailPolicy, RefusalTemplates
from agentic_rag.guardrails.refusal import render_refusal


def test_render_refusal_input_pii() -> None:
    """render_refusal: INPUT_PII reason uses input_pii template."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="ssn", start=0, end=11),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=20, end=40),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_PII, applied)

    assert "blocked" in result.lower()
    assert "ssn" in result.lower() or "email" in result.lower()


def test_render_refusal_input_injection() -> None:
    """render_refusal: INPUT_INJECTION reason uses input_injection template."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(
                detector="injection", entity="instruction_override", start=0, end=5
            ),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_INJECTION, applied)

    assert "blocked" in result.lower()
    assert "injection" in result.lower()


def test_render_refusal_output_pii() -> None:
    """render_refusal: OUTPUT_PII reason uses output_pii template."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="phone", start=10, end=22),
            action=Action.REDACT,
        ),
    ]

    result = render_refusal(policy, RefusalReason.OUTPUT_PII, applied)

    assert "withheld" in result.lower() or "answer" in result.lower()


def test_render_refusal_out_of_corpus_raises_value_error() -> None:
    """render_refusal: OUT_OF_CORPUS reason raises ValueError."""
    policy = GuardrailPolicy()

    with pytest.raises(ValueError, match="OUT_OF_CORPUS"):
        render_refusal(policy, RefusalReason.OUT_OF_CORPUS, [])


def test_render_refusal_dedupes_entities() -> None:
    """render_refusal: duplicate entities are deduplicated in output."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=0, end=10),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=20, end=30),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_PII, applied)

    # Should only have "email" once in the output
    assert result.count("email") == 1


def test_render_refusal_sorted_entities() -> None:
    """render_refusal: entities are sorted in output."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="ssn", start=0, end=5),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=10, end=20),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_PII, applied)

    # email should come before ssn alphabetically
    email_idx = result.lower().find("email")
    ssn_idx = result.lower().find("ssn")
    assert email_idx < ssn_idx


def test_render_refusal_sorted_categories() -> None:
    """render_refusal: injection categories are sorted in output."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="injection", entity="role_play", start=0, end=5),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(
                detector="injection", entity="instruction_override", start=10, end=20
            ),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_INJECTION, applied)

    # instruction_override should come before role_play alphabetically
    override_idx = result.lower().find("instruction_override")
    roleplay_idx = result.lower().find("role_play")
    assert override_idx < roleplay_idx


def test_render_refusal_custom_template() -> None:
    """render_refusal: custom template in policy renders correctly."""
    custom_policy = GuardrailPolicy(
        templates=RefusalTemplates(
            input_pii="Custom input PII: {entities}",
            input_injection="Custom injection: {categories}",
            output_pii="Custom output PII: {entities}",
        )
    )
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=0, end=10),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(custom_policy, RefusalReason.INPUT_PII, applied)

    assert result == "Custom input PII: email"


def test_render_refusal_multiple_entity_types() -> None:
    """render_refusal: multiple different entity types are comma-joined."""
    policy = GuardrailPolicy()
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=0, end=10),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(detector="regex", entity="phone", start=20, end=30),
            action=Action.BLOCK,
        ),
        AppliedDetection(
            detection=Detection(detector="regex", entity="ssn", start=40, end=50),
            action=Action.BLOCK,
        ),
    ]

    result = render_refusal(policy, RefusalReason.INPUT_PII, applied)

    # Should have comma-separated entities
    assert "," in result
    assert "email" in result.lower()
    assert "phone" in result.lower()
    assert "ssn" in result.lower()


def test_render_refusal_empty_applied() -> None:
    """render_refusal: empty applied detections renders template with empty entities."""
    policy = GuardrailPolicy(
        templates=RefusalTemplates(
            input_pii="Entities: '{entities}'",
            input_injection="Categories: '{categories}'",
            output_pii="Entities: '{entities}'",
        )
    )

    result = render_refusal(policy, RefusalReason.INPUT_PII, [])

    assert "Entities: ''" in result
