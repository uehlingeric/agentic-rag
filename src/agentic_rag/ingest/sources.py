"""NIST corpus registry with official URLs and verified checksums."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    """Registry entry for a corpus document."""

    doc_id: str
    title: str
    url: str
    sha256: str | None = None


SOURCES: dict[str, SourceDoc] = {
    "sp800-53r5": SourceDoc(
        doc_id="sp800-53r5",
        title="NIST SP 800-53 Rev. 5 (Security and Privacy Controls)",
        url="https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        sha256="fc63bcd61715d0181dd8e85998b1e6201ae3515fc6626102101cab1841e11ec6",
    ),
    "sp800-171r3": SourceDoc(
        doc_id="sp800-171r3",
        title="NIST SP 800-171 Rev. 3",
        url="https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r3.pdf",
        sha256="3e4631df8b5d61f40a6e542b52779ef30ddbbfff31e09214fa94ad6e6f5e6d08",
    ),
    "ai-rmf": SourceDoc(
        doc_id="ai-rmf",
        title="NIST AI RMF 1.0 (NIST.AI.100-1)",
        url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        sha256="7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1",
    ),
    "fips-199": SourceDoc(
        doc_id="fips-199",
        title="FIPS 199",
        url="https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.199.pdf",
        sha256="73d19f05f71e30f378050f178aa3943c38790bbae56c07f2b5708c5a1a90242f",
    ),
    "fips-200": SourceDoc(
        doc_id="fips-200",
        title="FIPS 200",
        url="https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.200.pdf",
        sha256="107a9b9cdc8eccf37386aec28bbaf2dfaef5cfece17151ceabc3960423243a57",
    ),
}
