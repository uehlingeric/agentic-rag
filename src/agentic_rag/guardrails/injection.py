"""Prompt-injection heuristic scanner (ADR-008, week 6).

Threat model: query injection (attacker controls the user question) and corpus
poisoning (attacker embeds a payload in a document that gets retrieved).

Posture: heuristics are a mitigation, not a solve. The patterns screen for
common attack vectors (instruction overrides, role-play, context-escape,
encoded payloads) but miss multilingual attacks, homoglyph/leetspeak
obfuscation, soft social-engineering instructions, and payload splitting.
The red-team suite (evals/redteam/attacks_v1.jsonl) contains annotated cases
of known misses (expect_catch=false) so published catch rates stay honest.

Documented false-positive tradeoffs:
- ``system:`` fires only at a line start; mid-sentence prose like
  "The system: security categorization ..." does not fire.
- Hex runs of 48+ chars fire, so SHA-256 digests quoted in security prose
  are flagged. A 48+ char hex run is rare enough in queries to accept.
- ``developer mode`` fires even in benign software-docs contexts.
- Base64 candidates must contain a digit and a letter and at most two ``/``
  characters: URL paths share the base64 alphabet but carry many slashes,
  while real base64 averages about one per 64 chars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_rag.guardrails.base import Detection, InjectionCategory


@dataclass(frozen=True, slots=True)
class _Match:
    """Internal: one pattern hit."""

    start: int
    end: int
    category: InjectionCategory


_OVERRIDE_PATTERNS = (
    # "ignore/disregard/forget ... instructions/prompts/rules/context" with
    # flexible wording between verb and object ([\w\s] cannot cross a period,
    # so a trigger split across sentences is a documented miss).
    re.compile(
        r"\b(ignore|disregard|forget)[\w\s]+(instructions|prompts|rules|context)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnew\s+(instructions|rules|prompts):", re.IGNORECASE),
    re.compile(r"\byour\s+(real|true|actual)\s+(instructions|prompts|rules)\b", re.IGNORECASE),
    re.compile(
        r"\b(reveal|print|show|repeat)\s+(?:your\s+)?(?:system\s+)?(prompt|instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\byou\s+must\s+now\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on[,]?\s+you\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+follow\b", re.IGNORECASE),
)

_ROLE_PLAY_PATTERNS = (
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(?:to\s+be|you\s+(?:are|have))\b", re.IGNORECASE),
    # "act as if/though" only: bare "act as" is ordinary prose in the NIST
    # corpus ("individuals who act as system administrators").
    re.compile(r"\bact\s+as\s+(?:if|though)\b", re.IGNORECASE),
    re.compile(r"\broleplay\s+as\b", re.IGNORECASE),
    # Uppercase DAN only (case-sensitive): the name "Dan" is ordinary prose.
    re.compile(r"\bDAN\b"),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:longer|more)\s+(?:bound|restricted|limited)\s+by\b", re.IGNORECASE),
)

_CONTEXT_ESCAPE_PATTERNS = (
    # Chat-template role markers
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"<<SYS>>|</s>", re.IGNORECASE),
    re.compile(r"###\s*(?:system|assistant|instruction)", re.IGNORECASE),
    re.compile(r"\bEND\s+OF\s+(?:CONTEXT|EXCERPTS|INSTRUCTIONS|PROMPT)\b", re.IGNORECASE),
    re.compile(r"\bBEGIN\s+(?:NEW\s+)?(?:SESSION|CONTEXT|CONVERSATION)\b", re.IGNORECASE),
    # Forged excerpt delimiters: queries never legitimately contain these, and
    # retrieved chunk text is neutralized upstream by the context builder.
    re.compile(r"</?excerpt", re.IGNORECASE),
    re.compile(r"^\s*system\s*:\s", re.IGNORECASE | re.MULTILINE),
)

_HEX_OR_ESCAPE_PATTERNS = (
    re.compile(r"\b(?:0x)?[0-9a-fA-F]{48,}\b"),
    re.compile(r"(?:\\u[0-9a-fA-F]{4}){4,}", re.IGNORECASE),
)

_BASE64_RUN = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

_PATTERNS: dict[InjectionCategory, tuple[re.Pattern[str], ...]] = {
    InjectionCategory.INSTRUCTION_OVERRIDE: _OVERRIDE_PATTERNS,
    InjectionCategory.ROLE_PLAY: _ROLE_PLAY_PATTERNS,
    InjectionCategory.CONTEXT_ESCAPE: _CONTEXT_ESCAPE_PATTERNS,
    InjectionCategory.ENCODED_PAYLOAD: _HEX_OR_ESCAPE_PATTERNS,
}


def _plausible_base64(run: str) -> bool:
    """Distinguish base64 payloads from long words and URL path segments."""
    return any(c.isdigit() for c in run) and any(c.isalpha() for c in run) and run.count("/") <= 2


class InjectionScanner:
    """Heuristic detector for prompt-injection attack patterns.

    Detects instruction overrides, role-play prompts, context-escape markers,
    and encoded payloads. This is a mitigation, not a solve — see the module
    docstring for known-miss classes and false-positive tradeoffs.
    """

    def scan(self, text: str) -> list[Detection]:
        """Scan text for injection patterns.

        Args:
            text: Input text to scan (query or retrieved chunk).

        Returns:
            Detections sorted by (start, end), deduplicated on exact
            (start, end, category). Detector is always "injection".
        """
        matches: list[_Match] = []
        for category, patterns in _PATTERNS.items():
            for pattern in patterns:
                for m in pattern.finditer(text):
                    matches.append(_Match(m.start(), m.end(), category))
        for m in _BASE64_RUN.finditer(text):
            if _plausible_base64(m.group()):
                matches.append(_Match(m.start(), m.end(), InjectionCategory.ENCODED_PAYLOAD))

        matches.sort(key=lambda m: (m.start, m.end))
        seen: set[tuple[int, int, InjectionCategory]] = set()
        detections: list[Detection] = []
        for match in matches:
            key = (match.start, match.end, match.category)
            if key in seen:
                continue
            seen.add(key)
            detections.append(
                Detection(
                    detector="injection",
                    entity=match.category,
                    start=match.start,
                    end=match.end,
                )
            )
        return detections
