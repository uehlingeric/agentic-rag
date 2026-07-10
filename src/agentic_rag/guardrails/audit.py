"""Audit logging: schema versioning and daily record rotation (ADR-008).

Schema versioning contract: audit_v1 fields are append-only. Breaking changes
bump the version. This decouples audit-consuming systems from deployment
velocity.

Privacy default: query hash not raw query (unless settings.guardrails.log_raw_query
is True). Audit records never include matched text from detections (the text
may BE the PII, and spans are useless without the text).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from agentic_rag.guardrails.base import AppliedDetection

AUDIT_SCHEMA = "audit_v1"


def sha256_hex(text: str) -> str:
    """SHA-256 hash of text (UTF-8 encoded) as hex string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """Summary of detections from one scan (input/output/retrieved).

    Spans are omitted from audit output: they are useless without the text,
    and the text may be PII. Matched text is never included. Detection
    records carry only the detector, entity type, and action taken.
    """

    detections: tuple[AppliedDetection, ...]
    blocked: bool

    def to_json(self) -> dict[str, object]:
        """Serialize to JSON-safe dict for audit record."""
        return {
            "detections": [
                {
                    "detector": d.detection.detector,
                    "entity": d.detection.entity,
                    "action": d.action,
                }
                for d in self.detections
            ],
            "blocked": self.blocked,
        }


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One complete audit record for a request-answer pair.

    All timestamps are UTC ISO 8601 (caller supplies). All text fields
    (query, answer, spans) are privacy-scrubbed per settings.guardrails
    log_raw_query flag.
    """

    request_id: str
    ts: str  # UTC ISO 8601
    query_sha256: str
    raw_query: str | None  # None unless log_raw_query is True
    provider: str
    model: str
    pipeline: str  # "vanilla" | "agentic"
    mode: str
    rerank: str
    guardrails_enabled: bool
    policy_version: int
    ner: bool
    input_scan: ScanSummary | None
    output_scan: ScanSummary | None
    retrieved_flagged_chunk_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]  # retrieved context chunk IDs
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: dict[str, float]
    refusal: bool
    refusal_reason: str | None
    answer_sha256: str | None  # None when blocked/refused with no answer


def record_to_json(record: AuditRecord) -> dict[str, object]:
    """Serialize AuditRecord to JSON-safe dict with schema marker.

    The schema field is always first, ensuring downstream systems can route
    by version.
    """
    data = asdict(record)

    # Convert nested dataclass to dict
    if record.input_scan is not None:
        data["input_scan"] = record.input_scan.to_json()
    if record.output_scan is not None:
        data["output_scan"] = record.output_scan.to_json()

    # Prepend schema
    return {"schema": AUDIT_SCHEMA, **data}


class AuditWriter:
    """Append audit records to daily rotated JSONL files.

    One file per day, named audit-YYYYMMDD.jsonl. Rotation follows record
    timestamp so tests are deterministic.
    """

    def __init__(self, audit_dir: Path) -> None:
        """Initialize writer for the audit directory.

        Args:
            audit_dir: Directory where audit-YYYYMMDD.jsonl files are written.
        """
        self.audit_dir = audit_dir

    def write(self, record: AuditRecord) -> Path:
        """Append one record to the daily audit file.

        Creates audit_dir and the file if needed. Returns the path written.

        Args:
            record: The AuditRecord to write.

        Returns:
            Path to the file appended to.
        """
        # Parse record timestamp to extract date for rotation
        dt = datetime.fromisoformat(record.ts)
        date_str = dt.strftime("%Y%m%d")
        filename = f"audit-{date_str}.jsonl"
        filepath = self.audit_dir / filename

        # Create directories if needed
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Append one JSON line
        json_line = json.dumps(record_to_json(record))
        with filepath.open("a") as f:
            f.write(json_line + "\n")

        return filepath
