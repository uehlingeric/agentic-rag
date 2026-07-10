"""Tests for audit logging and schema versioning."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.guardrails.audit import (
    AUDIT_SCHEMA,
    AuditRecord,
    AuditWriter,
    ScanSummary,
    record_to_json,
    sha256_hex,
)
from agentic_rag.guardrails.base import (
    Action,
    AppliedDetection,
    Detection,
)


def test_sha256_hex_known_vector() -> None:
    """sha256_hex: known vector 'abc' matches expected hash."""
    result = sha256_hex("abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert result == expected


def test_sha256_hex_empty_string() -> None:
    """sha256_hex: empty string produces valid hash."""
    result = sha256_hex("")
    # SHA-256 of empty string
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert result == expected


def test_scan_summary_to_json_no_detections() -> None:
    """ScanSummary.to_json: empty detections renders correctly."""
    summary = ScanSummary(detections=(), blocked=False)
    result = summary.to_json()

    assert result == {"detections": [], "blocked": False}


def test_scan_summary_to_json_with_detections() -> None:
    """ScanSummary.to_json: detections include detector, entity, action."""
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=0, end=10),
            action=Action.REDACT,
        ),
        AppliedDetection(
            detection=Detection(
                detector="injection",
                entity="instruction_override",
                start=20,
                end=30,
            ),
            action=Action.FLAG,
        ),
    ]
    summary = ScanSummary(detections=tuple(applied), blocked=False)
    result = summary.to_json()

    assert len(result["detections"]) == 2
    assert result["detections"][0] == {
        "detector": "regex",
        "entity": "email",
        "action": "redact",
    }
    assert result["detections"][1] == {
        "detector": "injection",
        "entity": "instruction_override",
        "action": "flag",
    }
    assert result["blocked"] is False


def test_scan_summary_to_json_no_spans_in_output() -> None:
    """ScanSummary.to_json: spans (start/end) are not in output."""
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=5, end=15),
            action=Action.REDACT,
        ),
    ]
    summary = ScanSummary(detections=tuple(applied), blocked=False)
    result = summary.to_json()

    # Spans should not be in the detection dict
    detection_dict = result["detections"][0]
    assert "start" not in detection_dict
    assert "end" not in detection_dict


def test_record_to_json_includes_schema() -> None:
    """record_to_json: output includes schema field first."""
    record = AuditRecord(
        request_id="req-123",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc123",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def456",
    )

    result = record_to_json(record)

    assert result["schema"] == AUDIT_SCHEMA
    # schema should be first key (ordering matters for parsing)
    keys = list(result.keys())
    assert keys[0] == "schema"


def test_record_to_json_no_matched_text() -> None:
    """record_to_json: serialized detections do not include matched text."""
    applied = [
        AppliedDetection(
            detection=Detection(detector="regex", entity="email", start=10, end=20),
            action=Action.REDACT,
        ),
    ]
    summary = ScanSummary(detections=tuple(applied), blocked=False)

    record = AuditRecord(
        request_id="req-123",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc123",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=summary,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def456",
    )

    result = record_to_json(record)

    # Check that input_scan has no "matched_text" or similar field
    input_scan = result["input_scan"]
    assert "matched_text" not in input_scan
    assert "text" not in input_scan
    # Detection should only have detector, entity, action
    detection = input_scan["detections"][0]
    assert set(detection.keys()) == {"detector", "entity", "action"}


def test_record_to_json_raw_query_when_provided() -> None:
    """record_to_json: raw_query is included when not None."""
    record = AuditRecord(
        request_id="req-123",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc123",
        raw_query="What is X?",  # Provided
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def456",
    )

    result = record_to_json(record)

    assert result["raw_query"] == "What is X?"


def test_record_to_json_raw_query_none() -> None:
    """record_to_json: raw_query can be None (default case)."""
    record = AuditRecord(
        request_id="req-123",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc123",
        raw_query=None,  # Privacy default
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def456",
    )

    result = record_to_json(record)

    assert result["raw_query"] is None


def test_audit_writer_writes_single_record(tmp_path: Path) -> None:
    """AuditWriter.write: appends one record to JSONL file."""
    audit_dir = tmp_path / "audit"
    writer = AuditWriter(audit_dir)

    record = AuditRecord(
        request_id="req-123",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc123",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def456",
    )

    path = writer.write(record)

    assert path.exists()
    assert path.name == "audit-20260710.jsonl"
    with path.open() as f:
        line = f.read().strip()
        data = json.loads(line)
        assert data["schema"] == AUDIT_SCHEMA
        assert data["request_id"] == "req-123"


def test_audit_writer_two_records_same_day_one_file(tmp_path: Path) -> None:
    """AuditWriter.write: two records on same day go to same file."""
    audit_dir = tmp_path / "audit"
    writer = AuditWriter(audit_dir)

    record1 = AuditRecord(
        request_id="req-1",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def",
    )

    record2 = AuditRecord(
        request_id="req-2",
        ts="2026-07-10T18:00:00Z",  # Same day
        query_sha256="xyz",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.002,
        latency_s={"total": 1.0},
        refusal=False,
        refusal_reason=None,
        answer_sha256="ghi",
    )

    path1 = writer.write(record1)
    path2 = writer.write(record2)

    # Should be same file
    assert path1 == path2
    assert path1.name == "audit-20260710.jsonl"

    # File should have two lines
    with path1.open() as f:
        lines = f.readlines()
        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["request_id"] == "req-1"
        assert data2["request_id"] == "req-2"


def test_audit_writer_different_days_different_files(tmp_path: Path) -> None:
    """AuditWriter.write: records on different days go to different files."""
    audit_dir = tmp_path / "audit"
    writer = AuditWriter(audit_dir)

    record1 = AuditRecord(
        request_id="req-1",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def",
    )

    record2 = AuditRecord(
        request_id="req-2",
        ts="2026-07-11T12:00:00Z",  # Different day
        query_sha256="xyz",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.002,
        latency_s={"total": 1.0},
        refusal=False,
        refusal_reason=None,
        answer_sha256="ghi",
    )

    path1 = writer.write(record1)
    path2 = writer.write(record2)

    # Should be different files
    assert path1 != path2
    assert path1.name == "audit-20260710.jsonl"
    assert path2.name == "audit-20260711.jsonl"


def test_audit_writer_creates_directories(tmp_path: Path) -> None:
    """AuditWriter.write: creates audit_dir if it doesn't exist."""
    audit_dir = tmp_path / "deep" / "audit" / "path"
    writer = AuditWriter(audit_dir)

    assert not audit_dir.exists()

    record = AuditRecord(
        request_id="req-1",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def",
    )

    path = writer.write(record)

    assert path.parent.exists()
    assert path.exists()


def test_audit_writer_returns_path(tmp_path: Path) -> None:
    """AuditWriter.write: returns the file path written."""
    audit_dir = tmp_path / "audit"
    writer = AuditWriter(audit_dir)

    record = AuditRecord(
        request_id="req-1",
        ts="2026-07-10T12:00:00Z",
        query_sha256="abc",
        raw_query=None,
        provider="anthropic",
        model="claude-sonnet",
        pipeline="vanilla",
        mode="default",
        rerank="none",
        guardrails_enabled=True,
        policy_version=1,
        ner=False,
        input_scan=None,
        output_scan=None,
        retrieved_flagged_chunk_ids=(),
        chunk_ids=(),
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_s={"total": 0.5},
        refusal=False,
        refusal_reason=None,
        answer_sha256="def",
    )

    path = writer.write(record)

    assert isinstance(path, Path)
    assert path.exists()
