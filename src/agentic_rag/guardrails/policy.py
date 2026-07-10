"""Policy configuration: load guardrails from YAML or defaults (ADR-008).

Policy-as-config: conservative federal posture is built into the code as
defaults; guardrails.yaml overrides them. This decouples policy evolution
from deployment mechanics and keeps tests hermetic (no filesystem reads
required). YAML is a deployment concern, not a test concern.

Flow per scanned text: detection (regex/NER/injection) → policy decision
(block/redact/flag) → audit or refusal. The policy file is the contract
between development and operations: deployments can tighten or loosen it
without code changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from agentic_rag.guardrails.base import (
    Action,
    AppliedDetection,
    Detection,
    PIIEntity,
    Verdict,
    redact,
)


class PIIPolicy(BaseModel):
    """PII handling: input and output each map entity types to actions."""

    model_config = ConfigDict(extra="forbid")

    input: dict[PIIEntity, Action] = {
        PIIEntity.SSN: Action.BLOCK,
        PIIEntity.EIN: Action.BLOCK,
        PIIEntity.CREDIT_CARD: Action.BLOCK,
        PIIEntity.PHONE: Action.REDACT,
        PIIEntity.EMAIL: Action.REDACT,
        PIIEntity.IP: Action.FLAG,
        PIIEntity.PERSON: Action.FLAG,
        PIIEntity.ORG: Action.FLAG,
    }

    output: dict[PIIEntity, Action] = {
        PIIEntity.SSN: Action.REDACT,
        PIIEntity.EIN: Action.REDACT,
        PIIEntity.CREDIT_CARD: Action.REDACT,
        PIIEntity.PHONE: Action.REDACT,
        PIIEntity.EMAIL: Action.REDACT,
        PIIEntity.IP: Action.FLAG,
        PIIEntity.PERSON: Action.FLAG,
        PIIEntity.ORG: Action.FLAG,
    }


class InjectionPolicy(BaseModel):
    """Injection handling: actions for user queries and retrieved corpus."""

    model_config = ConfigDict(extra="forbid")

    input: Action = Action.BLOCK
    retrieved: Action = Action.FLAG


class RefusalTemplates(BaseModel):
    """Template strings for refusal messages, using {entities}/{categories}
    str.format placeholders."""

    model_config = ConfigDict(extra="forbid")

    input_pii: str = (
        "This request was blocked by the input guardrail: it appears to "
        "contain {entities}. Remove the sensitive value(s) and ask again."
    )
    input_injection: str = (
        "This request was blocked by the input guardrail: it matches "
        "prompt-injection patterns ({categories})."
    )
    output_pii: str = (
        "The generated answer was withheld by the output guardrail: it contained {entities}."
    )


class GuardrailPolicy(BaseModel):
    """Complete guardrail policy: PII, injection, and refusal templates."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    pii: PIIPolicy = PIIPolicy()
    injection: InjectionPolicy = InjectionPolicy()
    templates: RefusalTemplates = RefusalTemplates()


def default_policy() -> GuardrailPolicy:
    """Built-in conservative guardrail policy (federal posture)."""
    return GuardrailPolicy()


def load_policy(path: Path | None) -> GuardrailPolicy:
    """Load policy from YAML file or return built-in defaults.

    Args:
        path: Path to guardrails.yaml. None → default_policy().

    Returns:
        Parsed and validated GuardrailPolicy.

    Raises:
        FileNotFoundError: If path is given but file does not exist.
        yaml.YAMLError: If YAML is malformed.
        pydantic.ValidationError: If policy violates schema (including
            unknown keys via extra="forbid").
    """
    if path is None:
        return default_policy()

    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    return GuardrailPolicy.model_validate(raw)


def apply_policy(
    policy: GuardrailPolicy,
    text: str,
    detections: list[Detection],
    *,
    direction: str,
) -> Verdict:
    """Apply policy to detections and redact text if needed.

    Direction determines which action a detection maps to:
    - "input": user query (before pipeline)
    - "output": model answer (after critic)
    - "retrieved": corpus text (trusted NIST, detection is for audit)

    Block wins: if any detection maps to BLOCK, the entire verdict is
    blocked=True, text unchanged. Otherwise, redact those with REDACT
    action and return the rewritten text.

    For PII detections: direction="input"|"output" consult policy.pii;
    direction="retrieved" always uses FLAG (corpus is public).

    For injection detections: direction="input" uses policy.injection.input;
    direction="retrieved" uses policy.injection.retrieved; direction="output"
    always uses FLAG (injection in our own output is audit-only).

    Args:
        policy: The active GuardrailPolicy.
        text: The scanned text.
        detections: All detections from PII/injection scanners.
        direction: One of "input", "output", "retrieved".

    Returns:
        Verdict with applied detections, blocked flag, and possibly
        redacted text.
    """
    applied: list[AppliedDetection] = []
    blocked = False

    for detection in detections:
        action: Action | None = None

        if detection.detector in ("regex", "ner"):
            # PII detection
            entity = PIIEntity(detection.entity)
            if direction == "retrieved":
                # Corpus is public; always flag
                action = Action.FLAG
            elif direction == "input":
                action = policy.pii.input.get(entity)
            elif direction == "output":
                action = policy.pii.output.get(entity)
        elif detection.detector == "injection":
            # Injection detection
            if direction == "input":
                action = policy.injection.input
            elif direction == "retrieved":
                action = policy.injection.retrieved
            elif direction == "output":
                action = Action.FLAG

        if action is None:
            # Unknown detector or direction; skip
            continue

        applied.append(AppliedDetection(detection=detection, action=action))
        if action == Action.BLOCK:
            blocked = True

    # Redact non-blocked detections
    redacted_text = text
    if not blocked:
        redact_detections = [d.detection for d in applied if d.action == Action.REDACT]
        redacted_text = redact(text, redact_detections)

    return Verdict(applied=tuple(applied), blocked=blocked, text=redacted_text)
