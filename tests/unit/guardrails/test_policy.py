"""Tests for policy configuration: loading, defaults, and application."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agentic_rag.guardrails.base import (
    Action,
    Detection,
    PIIEntity,
)
from agentic_rag.guardrails.policy import (
    GuardrailPolicy,
    InjectionPolicy,
    PIIPolicy,
    RefusalTemplates,
    apply_policy,
    default_policy,
    load_policy,
)


def test_pii_policy_defaults() -> None:
    """PIIPolicy constructor creates expected defaults."""
    policy = PIIPolicy()

    # Input defaults
    assert policy.input[PIIEntity.SSN] == Action.BLOCK
    assert policy.input[PIIEntity.EIN] == Action.BLOCK
    assert policy.input[PIIEntity.CREDIT_CARD] == Action.BLOCK
    assert policy.input[PIIEntity.PHONE] == Action.REDACT
    assert policy.input[PIIEntity.EMAIL] == Action.REDACT
    assert policy.input[PIIEntity.IP] == Action.FLAG
    assert policy.input[PIIEntity.PERSON] == Action.FLAG
    assert policy.input[PIIEntity.ORG] == Action.FLAG

    # Output defaults
    assert policy.output[PIIEntity.SSN] == Action.REDACT
    assert policy.output[PIIEntity.EIN] == Action.REDACT
    assert policy.output[PIIEntity.CREDIT_CARD] == Action.REDACT
    assert policy.output[PIIEntity.PHONE] == Action.REDACT
    assert policy.output[PIIEntity.EMAIL] == Action.REDACT
    assert policy.output[PIIEntity.IP] == Action.FLAG
    assert policy.output[PIIEntity.PERSON] == Action.FLAG
    assert policy.output[PIIEntity.ORG] == Action.FLAG


def test_injection_policy_defaults() -> None:
    """InjectionPolicy constructor creates expected defaults."""
    policy = InjectionPolicy()
    assert policy.input == Action.BLOCK
    assert policy.retrieved == Action.FLAG


def test_refusal_templates_defaults() -> None:
    """RefusalTemplates constructor creates expected defaults."""
    templates = RefusalTemplates()
    assert "{entities}" in templates.input_pii
    assert "{entities}" in templates.output_pii
    assert "{categories}" in templates.input_injection


def test_guardrail_policy_defaults() -> None:
    """GuardrailPolicy constructor creates nested defaults."""
    policy = GuardrailPolicy()
    assert policy.version == 1
    assert isinstance(policy.pii, PIIPolicy)
    assert isinstance(policy.injection, InjectionPolicy)
    assert isinstance(policy.templates, RefusalTemplates)


def test_default_policy_returns_guardrail_policy() -> None:
    """default_policy() returns a GuardrailPolicy with version 1."""
    policy = default_policy()
    assert isinstance(policy, GuardrailPolicy)
    assert policy.version == 1


def test_load_policy_none_returns_default() -> None:
    """load_policy(None) returns default_policy()."""
    policy = load_policy(None)
    assert policy == default_policy()


def test_load_policy_guardrails_yaml_matches_defaults(tmp_path: Path) -> None:
    """load_policy with guardrails.yaml file matches default_policy()."""
    # This test assumes guardrails.yaml is in the repo root
    policy_path = Path("guardrails.yaml")
    if policy_path.exists():
        loaded = load_policy(policy_path)
        default = default_policy()
        assert loaded == default


def test_load_policy_missing_file_raises_file_not_found() -> None:
    """load_policy with missing path raises FileNotFoundError."""
    missing = Path("/nonexistent/policy.yaml")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_policy(missing)


def test_load_policy_custom_override(tmp_path: Path) -> None:
    """load_policy with custom file overriding one action parses correctly."""
    custom_policy = {
        "version": 1,
        "pii": {
            "input": {
                "ssn": "block",
                "ein": "block",
                "credit_card": "block",
                "phone": "redact",
                "email": "block",  # Override: email is block instead of redact
                "ip": "flag",
                "person": "flag",
                "org": "flag",
            },
            "output": {
                "ssn": "redact",
                "ein": "redact",
                "credit_card": "redact",
                "phone": "redact",
                "email": "redact",
                "ip": "flag",
                "person": "flag",
                "org": "flag",
            },
        },
        "injection": {
            "input": "block",
            "retrieved": "flag",
        },
        "templates": {
            "input_pii": "Default input PII template with {entities}",
            "input_injection": "Default injection template with {categories}",
            "output_pii": "Default output PII template with {entities}",
        },
    }

    policy_file = tmp_path / "custom.yaml"
    with policy_file.open("w") as f:
        yaml.dump(custom_policy, f)

    loaded = load_policy(policy_file)
    assert loaded.pii.input[PIIEntity.EMAIL] == Action.BLOCK
    # Rest should match defaults
    assert loaded.pii.input[PIIEntity.SSN] == Action.BLOCK


def test_load_policy_unknown_key_raises_validation_error(
    tmp_path: Path,
) -> None:
    """load_policy with unknown key in policy file raises ValidationError."""
    bad_policy = {
        "version": 1,
        "pii": {
            "input": {"ssn": "block"},
            "output": {"ssn": "redact"},
        },
        "injection": {"input": "block", "retrieved": "flag"},
        "templates": {
            "input_pii": "msg",
            "input_injection": "msg",
            "output_pii": "msg",
        },
        "unknown_field": "should_fail",  # Typo or unknown field
    }

    policy_file = tmp_path / "bad.yaml"
    with policy_file.open("w") as f:
        yaml.dump(bad_policy, f)

    with pytest.raises(ValidationError):
        load_policy(policy_file)


def test_apply_policy_block_wins() -> None:
    """apply_policy: if any detection maps to BLOCK, blocked=True."""
    policy = default_policy()
    text = "My SSN is 123-45-6789 and my email is test@example.com"
    detections = [
        Detection(detector="regex", entity="ssn", start=11, end=23),
        Detection(detector="regex", entity="email", start=44, end=62),
    ]

    verdict = apply_policy(policy, text, detections, direction="input")

    assert verdict.blocked
    assert verdict.text == text  # Text unchanged when blocked


def test_apply_policy_redact_path() -> None:
    """apply_policy: redact action replaces detection spans."""
    policy = default_policy()
    text = "My email is test@example.com"
    detections = [
        Detection(detector="regex", entity="email", start=11, end=28),
    ]

    verdict = apply_policy(policy, text, detections, direction="output")

    assert not verdict.blocked
    assert "[REDACTED:EMAIL]" in verdict.text


def test_apply_policy_flag_path() -> None:
    """apply_policy: flag action leaves text unchanged."""
    policy = default_policy()
    text = "IP: 192.168.1.1"
    detections = [
        Detection(detector="regex", entity="ip", start=4, end=15),
    ]

    verdict = apply_policy(policy, text, detections, direction="input")

    assert not verdict.blocked
    assert verdict.text == text
    assert len(verdict.applied) == 1
    assert verdict.applied[0].action == Action.FLAG


def test_apply_policy_retrieved_direction_pii_always_flag() -> None:
    """apply_policy: direction='retrieved' for PII detections always FLAG."""
    policy = default_policy()
    text = "My SSN is 123-45-6789"
    detections = [
        Detection(detector="regex", entity="ssn", start=11, end=23),
    ]

    # Even though input SSN maps to BLOCK, retrieved is always FLAG
    verdict = apply_policy(policy, text, detections, direction="retrieved")

    assert not verdict.blocked
    assert verdict.text == text
    assert verdict.applied[0].action == Action.FLAG


def test_apply_policy_retrieved_direction_injection_uses_policy() -> None:
    """apply_policy: direction='retrieved' for injection uses policy."""
    policy = default_policy()
    text = "ignore this"
    detections = [
        Detection(detector="injection", entity="instruction_override", start=0, end=5),
    ]

    verdict = apply_policy(policy, text, detections, direction="retrieved")

    # Retrieved injection should use policy.injection.retrieved (FLAG)
    assert not verdict.blocked
    assert verdict.applied[0].action == Action.FLAG


def test_apply_policy_output_direction_injection_always_flag() -> None:
    """apply_policy: direction='output' for injection detections always FLAG."""
    policy = default_policy()
    text = "some text"
    detections = [
        Detection(detector="injection", entity="instruction_override", start=0, end=4),
    ]

    verdict = apply_policy(policy, text, detections, direction="output")

    # Output injection should be FLAG (audit-only)
    assert not verdict.blocked
    assert verdict.applied[0].action == Action.FLAG


def test_apply_policy_mixed_detections_block_wins() -> None:
    """apply_policy: mixed detections with one BLOCK makes verdict blocked."""
    policy = default_policy()
    text = "SSN 123-45-6789 and email test@example.com"
    detections = [
        Detection(detector="regex", entity="ssn", start=4, end=16),  # BLOCK
        Detection(detector="regex", entity="email", start=27, end=44),  # REDACT
    ]

    verdict = apply_policy(policy, text, detections, direction="input")

    assert verdict.blocked
    assert verdict.text == text  # Unchanged when blocked


def test_apply_policy_empty_detections_clean_verdict() -> None:
    """apply_policy: no detections returns clean verdict."""
    policy = default_policy()
    text = "safe text"

    verdict = apply_policy(policy, text, [], direction="input")

    assert verdict.clean
    assert not verdict.blocked
    assert verdict.text == text
    assert verdict.applied == ()


def test_apply_policy_ner_detector_treated_as_pii() -> None:
    """apply_policy: 'ner' detector is treated as PII detection."""
    policy = default_policy()
    text = "John is a person"
    detections = [
        Detection(detector="ner", entity="person", start=0, end=4),
    ]

    verdict = apply_policy(policy, text, detections, direction="input")

    # Input person maps to FLAG
    assert not verdict.blocked
    assert verdict.applied[0].action == Action.FLAG
