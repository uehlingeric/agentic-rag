"""Tests for PII scanner: regex layer, NER layer, span integrity, and policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from agentic_rag.guardrails.base import PIIEntity, redact
from agentic_rag.guardrails.pii import (
    NERDependencyError,
    PIIScanner,
    _luhn,
)

# Load fixtures from JSONL at collection time
_fixture_path = Path(__file__).parent / "fixtures" / "pii_cases.jsonl"
_pii_cases = []
with open(_fixture_path) as f:
    for line in f:
        line = line.strip()
        if line:
            _pii_cases.append(json.loads(line))


# Parametrized test over fixtures
@pytest.mark.parametrize("case", _pii_cases, ids=lambda c: c["id"])
def test_pii_detection_fixture(case: dict) -> None:
    """Test PII detection against fixture cases."""
    text = case["text"]
    expected = case["expect"]

    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    # Count detections by entity type
    actual_counts = {}
    for d in detections:
        entity = d.entity.lower() if isinstance(d.entity, str) else str(d.entity)
        actual_counts[entity] = actual_counts.get(entity, 0) + 1

    expected_counts = {}
    for e in expected:
        entity = e["entity"].lower()
        expected_counts[entity] = expected_counts.get(entity, 0) + e.get("count", 1)

    assert actual_counts == expected_counts, (
        f"Case {case['id']}: expected {expected_counts}, got {actual_counts}"
    )


def test_span_integrity() -> None:
    """Test that detection spans are valid and non-empty."""
    text = "Contact alice@example.com or call 703-555-0100"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    for detection in detections:
        # Span must be within text bounds
        assert 0 <= detection.start < len(text), f"Invalid start: {detection.start}"
        assert 0 < detection.end <= len(text), f"Invalid end: {detection.end}"
        # Span must be non-empty
        assert detection.start < detection.end, "Empty span"
        # Text at span must be non-empty
        span_text = text[detection.start : detection.end]
        assert len(span_text) > 0, "Span extracts empty string"


def test_sorted_output() -> None:
    """Test that detections are sorted by (start, end)."""
    text = "Email: alice@example.com Phone: 703-555-0100"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    spans = [(d.start, d.end) for d in detections]
    sorted_spans = sorted(spans)
    assert spans == sorted_spans, f"Unsorted output: {spans}"


def test_detector_labels() -> None:
    """Test that detector labels are correct."""
    text = "Contact: alice@example.com"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    # Regex-only scanner should only have regex detections
    for d in detections:
        assert d.detector == "regex", f"Expected regex, got {d.detector}"


def test_luhn_valid() -> None:
    """Test Luhn validation for a valid card number."""
    # 4111111111111111 is a valid test card (Visa)
    assert _luhn("4111111111111111") is True


def test_luhn_invalid() -> None:
    """Test Luhn validation rejects invalid numbers."""
    # 4111111111111112 is invalid (off by one)
    assert _luhn("4111111111111112") is False


def test_luhn_invalid_16_digits() -> None:
    """Test Luhn validation with a 16-digit invalid number."""
    assert _luhn("4111111111111113") is False


def test_ssn_dash_format() -> None:
    """Test SSN with dash separators."""
    text = "SSN: 123-45-6789"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ssn_detections = [d for d in detections if d.entity == PIIEntity.SSN]
    assert len(ssn_detections) == 1
    assert text[ssn_detections[0].start : ssn_detections[0].end] == "123-45-6789"


def test_ssn_dot_format() -> None:
    """Test SSN with dot separators."""
    text = "SSN: 123.45.6789"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ssn_detections = [d for d in detections if d.entity == PIIEntity.SSN]
    assert len(ssn_detections) == 1
    assert text[ssn_detections[0].start : ssn_detections[0].end] == "123.45.6789"


def test_ssn_space_format() -> None:
    """Test SSN with space separators."""
    text = "SSN: 123 45 6789"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ssn_detections = [d for d in detections if d.entity == PIIEntity.SSN]
    assert len(ssn_detections) == 1
    assert text[ssn_detections[0].start : ssn_detections[0].end] == "123 45 6789"


def test_ssn_not_bare_nine_digits() -> None:
    """Test that bare 9-digit runs do NOT match (they're document IDs)."""
    text = "Document 123456789 is classified."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ssn_detections = [d for d in detections if d.entity == PIIEntity.SSN]
    assert len(ssn_detections) == 0, "Bare 9-digit run should not match"


def test_ein_format() -> None:
    """Test EIN detection."""
    text = "EIN: 12-3456789"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ein_detections = [d for d in detections if d.entity == PIIEntity.EIN]
    assert len(ein_detections) == 1
    assert text[ein_detections[0].start : ein_detections[0].end] == "12-3456789"


def test_phone_with_area_code_parentheses() -> None:
    """Test NANP phone with (area) format."""
    text = "Call (703) 555-0100 for support."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 1


def test_phone_with_dashes() -> None:
    """Test NANP phone with dashes."""
    text = "Call 703-555-0100 now."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 1


def test_phone_with_dots() -> None:
    """Test NANP phone with dots."""
    text = "Call 703.555.0100 now."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 1


def test_phone_with_plus_one() -> None:
    """Test NANP phone with +1 prefix."""
    text = "Call +1 703 555 0100 from abroad."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 1


def test_phone_hard_negative_nist_sp_800_53() -> None:
    """Test that 'SP 800-53' does NOT match (control ID, not phone)."""
    text = "SP 800-53 defines security controls."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 0, "SP 800-53 must not match as phone"


def test_phone_hard_negative_800_171() -> None:
    """Test that '800-171' does NOT match (standard number, not phone)."""
    text = "NIST 800-171 is a popular standard."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 0, "800-171 must not match as phone"


def test_phone_hard_negative_section_number() -> None:
    """Test that section '3.13.11' does NOT match (section number, not phone)."""
    text = "See section 3.13.11 for details."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]
    assert len(phone_detections) == 0, "Section numbers must not match"


def test_email_standard_format() -> None:
    """Test standard email address."""
    text = "Contact alice@example.com for support."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    email_detections = [d for d in detections if d.entity == PIIEntity.EMAIL]
    assert len(email_detections) >= 1, "Standard email should match"


def test_email_obfuscated_at_dot() -> None:
    """Test obfuscated email with [at] and [dot]."""
    text = "Email: alice [at] example [dot] com"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    email_detections = [d for d in detections if d.entity == PIIEntity.EMAIL]
    assert len(email_detections) >= 1, "Obfuscated [at]/[dot] should match"


def test_credit_card_valid_visa() -> None:
    """Test valid Visa card (passes Luhn)."""
    text = "Card: 4111111111111111"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    cc_detections = [d for d in detections if d.entity == PIIEntity.CREDIT_CARD]
    assert len(cc_detections) == 1


def test_credit_card_invalid_luhn() -> None:
    """Test that invalid Luhn number does NOT match."""
    text = "Card: 4111111111111112"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    cc_detections = [d for d in detections if d.entity == PIIEntity.CREDIT_CARD]
    assert len(cc_detections) == 0, "Invalid Luhn should not match"


def test_credit_card_with_dashes() -> None:
    """Test credit card with dashes."""
    text = "4111-1111-1111-1111"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    cc_detections = [d for d in detections if d.entity == PIIEntity.CREDIT_CARD]
    assert len(cc_detections) == 1


def test_credit_card_with_spaces() -> None:
    """Test credit card with spaces."""
    text = "4111 1111 1111 1111"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    cc_detections = [d for d in detections if d.entity == PIIEntity.CREDIT_CARD]
    assert len(cc_detections) == 1


def test_ip_valid_ipv4() -> None:
    """Test valid IPv4 detection."""
    text = "Server at 192.168.1.1 is online."
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ip_detections = [d for d in detections if d.entity == PIIEntity.IP]
    assert len(ip_detections) == 1


def test_ip_edge_case_0_0_0_0() -> None:
    """Test IPv4 0.0.0.0."""
    text = "Broadcast from 0.0.0.0"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ip_detections = [d for d in detections if d.entity == PIIEntity.IP]
    assert len(ip_detections) == 1


def test_ip_edge_case_255_255_255_255() -> None:
    """Test IPv4 255.255.255.255."""
    text = "Broadcast to 255.255.255.255"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ip_detections = [d for d in detections if d.entity == PIIEntity.IP]
    assert len(ip_detections) == 1


def test_ip_invalid_octet_too_large() -> None:
    """Test that IPv4 with octet > 255 does NOT match."""
    text = "Invalid IP: 999.1.1.1"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ip_detections = [d for d in detections if d.entity == PIIEntity.IP]
    assert len(ip_detections) == 0, "Invalid octet should not match"


def test_ip_hard_negative_version_string() -> None:
    """Test that version string like '10.0.19041.1' does NOT match."""
    text = "Windows version 10.0.19041.1"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    ip_detections = [d for d in detections if d.entity == PIIEntity.IP]
    # Should not match 19041.1 because 19041 > 255
    assert len(ip_detections) == 0, "Version strings must not match"


def test_multi_entity_email_and_phone() -> None:
    """Test text with both email and phone."""
    text = "Contact alice@example.com or call 703-555-0100"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    email_detections = [d for d in detections if d.entity == PIIEntity.EMAIL]
    phone_detections = [d for d in detections if d.entity == PIIEntity.PHONE]

    assert len(email_detections) >= 1, "Should detect email"
    assert len(phone_detections) >= 1, "Should detect phone"


def test_redaction_integration() -> None:
    """Test redaction of email and phone."""
    text = "Contact alice@example.com or call 703-555-0100"
    scanner = PIIScanner(ner=False)
    detections = scanner.scan(text)

    redacted = redact(text, detections)

    # Redacted text should contain [REDACTED:...] markers
    assert "[REDACTED:" in redacted, "Should contain redaction markers"
    # Original sensitive strings should be gone
    assert "alice@example.com" not in redacted
    assert "703-555-0100" not in redacted
    # Remaining prose should be intact
    assert "Contact" in redacted
    assert "or call" in redacted


def test_ner_dependency_missing_spacy() -> None:
    """Test NERDependencyError when spacy is not available."""
    # Monkeypatch sys.modules to simulate missing spacy
    with mock.patch.dict(sys.modules, {"spacy": None}):
        with pytest.raises(NERDependencyError) as exc_info:
            PIIScanner(ner=True)
        assert "guardrails-ner" in str(exc_info.value)


def test_ner_dependency_missing_model() -> None:
    """Test NERDependencyError when model en_core_web_sm is missing."""
    spacy_module = pytest.importorskip("spacy")
    # Monkeypatch spacy.load to raise OSError
    with mock.patch.object(spacy_module, "load", side_effect=OSError("Model not found")):
        with pytest.raises(NERDependencyError) as exc_info:
            PIIScanner(ner=True)
        assert "en_core_web_sm" in str(exc_info.value) or "guardrails-ner" in str(exc_info.value)


@pytest.mark.skipif(
    True,  # Skip unless explicitly testing NER
    reason="NER tests require spacy model; run separately if available",
)
def test_ner_person_detection() -> None:
    """Test NER detection of person names."""
    pytest.importorskip("spacy")
    try:
        scanner = PIIScanner(ner=True)
    except NERDependencyError:
        pytest.skip("spacy or model not available")

    text = "My name is John Smith and I work at Acme Corporation."
    detections = scanner.scan(text)

    ner_detections = [d for d in detections if d.detector == "ner"]
    # Should detect at least one of PERSON or ORG
    entities = {d.entity for d in ner_detections}
    assert PIIEntity.PERSON in entities or PIIEntity.ORG in entities


def test_ner_overlap_filtering() -> None:
    """Test that NER entities overlapping regex detections are filtered.

    This is a simplified test; a full test would require monkeypatching
    the NER pipeline to produce overlapping entities.
    """
    pytest.importorskip("spacy")
    try:
        scanner = PIIScanner(ner=True)
    except NERDependencyError:
        pytest.skip("spacy or model not available")

    # A text where regex might overlap NER (if NER fired)
    text = "Email: alice@example.com is valid."
    detections = scanner.scan(text)

    # Just verify the scan completes and produces sorted output
    spans = [(d.start, d.end) for d in detections]
    assert spans == sorted(spans)
