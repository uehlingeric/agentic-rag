"""Shared fixtures for retrieval tests. Owned by the coordinator — do not edit
in feature work; add module-specific fixtures in the test module itself."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.base import ChunkRecord


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    section_ids: list[str] | None = None,
    heading: str = "AC-2 ACCOUNT MANAGEMENT",
    text: str = "The organization manages system accounts.",
    token_count: int = 10,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=section_ids if section_ids is not None else [section_id],
        section_path=heading,
        heading=heading,
        page_start=1,
        page_end=2,
        token_count=token_count,
        text=text,
    )


@pytest.fixture()
def tiny_corpus() -> list[ChunkRecord]:
    """Five chunks with distinct vocabulary so expected rankings are obvious."""
    return [
        make_chunk(
            "c-access",
            section_id="AC-2",
            heading="AC-2 ACCOUNT MANAGEMENT",
            text=(
                "Account management requires the organization to define authorized "
                "users of the system, group and role membership, and access "
                "authorizations for each account."
            ),
        ),
        make_chunk(
            "c-audit",
            section_id="AU-2",
            heading="AU-2 EVENT LOGGING",
            text=(
                "Event logging identifies the types of events that the system is "
                "capable of logging in support of the audit function."
            ),
        ),
        make_chunk(
            "c-risk",
            doc_id="ai-rmf",
            section_id="GOVERN-1",
            heading="GOVERN 1",
            text=(
                "Legal and regulatory requirements involving artificial intelligence "
                "are understood, managed, and documented as part of risk governance."
            ),
        ),
        make_chunk(
            "c-crypto",
            doc_id="fips-199",
            section_id="3",
            heading="3 CATEGORIZATION",
            text=(
                "Security categorization standards provide a common framework for "
                "expressing confidentiality, integrity, and availability impact levels."
            ),
        ),
        make_chunk(
            "c-training",
            section_id="AT-2",
            heading="AT-2 LITERACY TRAINING AND AWARENESS",
            text=(
                "Literacy training and awareness ensures personnel receive security "
                "and privacy training before authorizing access to the system."
            ),
        ),
    ]
