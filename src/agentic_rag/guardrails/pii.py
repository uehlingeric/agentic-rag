"""PII detection via regex layer and optional spaCy NER.

The scanner operates on two detection layers:

1. Regex layer (always on): Detects structured PII patterns (SSN, EIN, phone,
   email, credit card, IPv4) via hand-crafted regex compiled at module load.
   Designed for high precision on the NIST security publications corpus,
   where false positives on control IDs, section numbers, and version strings
   must not occur. Non-detections (SSN bare 9-digit runs, PHONE bare 7-digit
   local numbers) are intentional: they are indistinguishable from document
   IDs and section numbers and honest tradeoffs given the corpus domain.

2. NER layer (optional, requires spacy + en_core_web_sm): Named entity
   recognition via a pretrained spaCy model to detect PERSON and ORG entities.
   Off by default; enable with ner=True. NER entities whose spans overlap an
   existing regex detection are filtered out to avoid double-reporting.

Both layers sort detections by (start, end) span and tag detector="regex"
or detector="ner" accordingly.

Policy logic (block/redact/flag) is not implemented here; it lives in
the policy layer. This module reports detections only.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentic_rag.guardrails.base import Detection, PIIEntity

if TYPE_CHECKING:
    import spacy


class NERDependencyError(RuntimeError):
    """Raised when ner=True but spacy or its model is unavailable."""

    pass


def _luhn(digits: str) -> bool:
    """Validate a credit card number via Luhn algorithm.

    Args:
        digits: String of digits only (no separators).

    Returns:
        True if the digits pass Luhn validation, False otherwise.
    """
    total = 0
    reverse_digits = digits[::-1]
    for i, char in enumerate(reverse_digits):
        digit = int(char)
        # Double every second digit (odd index in reversed string)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Compile regex patterns at module load time. Each pattern is designed
# with careful boundaries to avoid false positives in the NIST corpus.

# SSN: 3-2-4 digits with consistent separator (dash, dot, or space).
# Boundary logic: reject if adjacent chars are digits, dashes, or dots,
# so version-like contexts (1123-45-67890) don't fire.
# Do NOT detect bare 9-digit runs — they're indistinguishable from
# document/serial numbers and honest context-dependent tradeoff.
_SSN_PATTERN = re.compile(r"(?<![0-9./-])\d{3}([-.]|\s)\d{2}\1\d{4}(?![0-9./-])")

# EIN: 2-7 digits, word-bounded, not adjacent to digits/dashes.
_EIN_PATTERN = re.compile(r"\b\d{2}-\d{7}\b")

# PHONE: NANP 10-digit with visible structure to avoid false positives.
# Matches: (703) 555-0100, 703-555-0100, 703.555.0100, +1 703 555 0100.
# The unparenthesized branch requires one consistent separator (backreference)
# and boundary lookarounds, so bare 10-digit runs (credit cards) and short
# digit-dash runs never fire. Do NOT detect bare 7-digit local numbers
# (e.g., "555-0100"): NIST publication ids like "800-53" live in a world of
# short digit-dash runs.
_PHONE_PATTERN = re.compile(
    r"(?:"
    r"\(\d{3}\)[\s.-]?\d{3}[\s.-]\d{4}"  # (703) 555-0100
    r"|"
    r"(?<![\d.-])(?:\+?1[\s.-])?\d{3}([\s.-])\d{3}\1\d{4}"  # 703-555-0100, +1 703 555 0100
    r")(?![\d-])"
)

# EMAIL: Standard RFC-like pattern plus obfuscated forms.
# Obfuscated forms: "user [at] example [dot] com", "(at)", "(dot)" variants.
_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+\-]+(?:@|[\s]*\((?:at|@)\)[\s]*|[\s]*\[(?:at|@)\][\s]*)"
    r"[A-Za-z0-9.\-]+(?:(?:\.|[\s]*\((?:dot|\.)\)[\s]*|[\s]*\[(?:dot|\.)\][\s]*))"
    r"[A-Za-z]+",
    re.IGNORECASE,
)

# CREDIT_CARD: 13-19 digits optionally grouped by spaces/dashes.
# Luhn validation is performed in code; this regex just extracts candidates.
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[\s\-]?){12,18}\d\b")

# IPv4: Dotted quad with word boundary. Each octet must be 0-255
# (validated in code). Do NOT detect bare 5-digit groups like "19041"
# in "10.0.19041.1" version strings. Word boundary ensures safety.
# IPv6 is deliberately not detected: it's verbose, rarely appears in
# plain text, and spaCy doesn't reliably detect it anyway.
_IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)


class PIIScanner:
    """Scan text for PII using regex and optional NER."""

    def __init__(self, *, ner: bool = False) -> None:
        """Initialize the PII scanner.

        Args:
            ner: Enable optional spaCy NER layer. Requires spacy and
                en_core_web_sm model. If True but dependencies are missing,
                raises NERDependencyError.

        Raises:
            NERDependencyError: If ner=True and spacy or model is unavailable.
        """
        self._ner_enabled = ner
        self._ner_pipeline: spacy.Language | None = None
        if ner:
            try:
                import spacy
            except ImportError as e:
                raise NERDependencyError(
                    "spacy is not installed. Install it via:\n"
                    '  uv pip install -e ".[guardrails-ner]"\n'
                    "  uv run python -m spacy download en_core_web_sm"
                ) from e
            try:
                self._ner_pipeline = spacy.load("en_core_web_sm")
            except OSError as e:
                raise NERDependencyError(
                    "spacy model en_core_web_sm not found. Download it via:\n"
                    '  uv pip install -e ".[guardrails-ner]"\n'
                    "  uv run python -m spacy download en_core_web_sm"
                ) from e

    def scan(self, text: str) -> list[Detection]:
        """Scan text for PII detections.

        Returns detections sorted by (start, end) span.

        Args:
            text: Text to scan.

        Returns:
            List of Detection objects sorted by span start and end.
        """
        detections: list[Detection] = []

        # Regex layer: always on
        detections.extend(self._scan_regex(text))

        # NER layer: optional
        if self._ner_enabled and self._ner_pipeline is not None:
            detections.extend(self._scan_ner(text, detections))

        # Sort by (start, end)
        return sorted(detections, key=lambda d: (d.start, d.end))

    def _scan_regex(self, text: str) -> list[Detection]:
        """Scan text with regex patterns."""
        detections: list[Detection] = []

        # SSN
        for match in _SSN_PATTERN.finditer(text):
            detections.append(
                Detection(
                    detector="regex",
                    entity=PIIEntity.SSN,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # EIN
        for match in _EIN_PATTERN.finditer(text):
            detections.append(
                Detection(
                    detector="regex",
                    entity=PIIEntity.EIN,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # PHONE
        for match in _PHONE_PATTERN.finditer(text):
            detections.append(
                Detection(
                    detector="regex",
                    entity=PIIEntity.PHONE,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # EMAIL
        for match in _EMAIL_PATTERN.finditer(text):
            detections.append(
                Detection(
                    detector="regex",
                    entity=PIIEntity.EMAIL,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # CREDIT_CARD (with Luhn validation)
        for match in _CREDIT_CARD_PATTERN.finditer(text):
            # Extract digits only
            digits_only = "".join(c for c in match.group() if c.isdigit())
            if len(digits_only) in range(13, 20) and _luhn(digits_only):
                detections.append(
                    Detection(
                        detector="regex",
                        entity=PIIEntity.CREDIT_CARD,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # IP
        for match in _IP_PATTERN.finditer(text):
            detections.append(
                Detection(
                    detector="regex",
                    entity=PIIEntity.IP,
                    start=match.start(),
                    end=match.end(),
                )
            )

        return detections

    def _scan_ner(self, text: str, regex_detections: list[Detection]) -> list[Detection]:
        """Scan text with spaCy NER, filtering overlaps with regex.

        NER entities whose spans overlap a regex detection are skipped.

        Args:
            text: Text to scan.
            regex_detections: List of regex detections (for overlap filtering).

        Returns:
            List of NER detections not overlapping regex detections.
        """
        if self._ner_pipeline is None:
            return []

        detections: list[Detection] = []
        doc = self._ner_pipeline(text)

        # Build a set of regex spans for fast overlap checking
        regex_spans = {(d.start, d.end) for d in regex_detections}

        for ent in doc.ents:
            # Map spaCy label to PIIEntity
            if ent.label_ == "PERSON":
                entity = PIIEntity.PERSON
            elif ent.label_ == "ORG":
                entity = PIIEntity.ORG
            else:
                # Ignore other entity types
                continue

            # Skip if overlaps a regex detection
            if any(
                d_start < ent.end_char and ent.start_char < d_end for d_start, d_end in regex_spans
            ):
                continue

            detections.append(
                Detection(
                    detector="ner",
                    entity=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return detections
