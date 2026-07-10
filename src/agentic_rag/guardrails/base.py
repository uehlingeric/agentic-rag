"""Guardrail contracts: entities, actions, detections, verdicts (ADR-008).

Layering: guardrails may import pipeline contracts; pipeline modules never
import guardrails. The wire-through (``GuardedPipeline``) lives in
``agentic_rag.guardrails.guarded`` and wraps a pipeline from the outside.

Detections carry spans into the scanned text but never the matched text
itself — a detection may BE the PII, and detections flow into audit records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PIIEntity(StrEnum):
    """Entity types the PII scanner reports.

    ``ssn``..``ip`` come from the always-on regex layer; ``person`` and
    ``org`` come from the optional spaCy NER layer ([guardrails-ner] extra).
    """

    SSN = "ssn"
    EIN = "ein"
    PHONE = "phone"
    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    IP = "ip"
    PERSON = "person"
    ORG = "org"


class InjectionCategory(StrEnum):
    """Attack categories the injection heuristics screen for."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_PLAY = "role_play"
    CONTEXT_ESCAPE = "context_escape"
    ENCODED_PAYLOAD = "encoded_payload"


class Action(StrEnum):
    """Policy action for a detection.

    ``block`` refuses the request (input) or replaces the answer (output);
    ``redact`` substitutes ``[REDACTED:{ENTITY}]`` and proceeds; ``flag``
    proceeds unchanged and records the detection in the audit log.
    """

    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"


class RefusalReason(StrEnum):
    """Machine-readable ``Answer.refusal_reason`` values.

    ``out_of_corpus`` is the model's own grounded refusal (leading
    [NO_ANSWER]); the others are guardrail verdicts.
    """

    OUT_OF_CORPUS = "out_of_corpus"
    INPUT_PII = "input_pii"
    INPUT_INJECTION = "input_injection"
    OUTPUT_PII = "output_pii"


@dataclass(frozen=True, slots=True)
class Detection:
    """One scanner hit. ``detector`` is ``regex`` | ``ner`` | ``injection``;
    ``entity`` is a ``PIIEntity`` or ``InjectionCategory`` value. ``start``/
    ``end`` index into the scanned text (end exclusive)."""

    detector: str
    entity: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class AppliedDetection:
    """A detection paired with the policy action taken for it."""

    detection: Detection
    action: Action


@dataclass(frozen=True, slots=True)
class Verdict:
    """Outcome of applying policy to one scanned text.

    ``text`` is the post-policy text: redactions applied when any detection's
    action is ``redact``, otherwise the input unchanged. ``blocked`` is True
    when any detection's action is ``block`` — the caller must not use
    ``text`` downstream in that case.
    """

    applied: tuple[AppliedDetection, ...]
    blocked: bool
    text: str

    @property
    def clean(self) -> bool:
        """True when nothing was detected."""
        return not self.applied


def redact(text: str, detections: list[Detection]) -> str:
    """Replace detection spans with ``[REDACTED:{ENTITY}]`` markers.

    Overlapping spans are merged into one marker carrying the
    earliest-starting detection's entity label; replacement runs
    right-to-left over the merged spans so indices stay valid.
    """
    if not detections:
        return text
    spans = sorted((d.start, d.end, d.entity) for d in detections)
    merged: list[tuple[int, int, str]] = []
    for start, end, entity in spans:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_entity = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_entity)
        else:
            merged.append((start, end, entity))
    for start, end, entity in reversed(merged):
        text = text[:start] + f"[REDACTED:{entity.upper()}]" + text[end:]
    return text
