"""Fixtures for ingest tests."""

import pytest

from agentic_rag.ingest.extract import Section


@pytest.fixture
def small_section() -> Section:
    """A small section fitting in single chunk (~100 tokens)."""
    text = (
        "The quick brown fox jumps over the lazy dog. This is a short section. "
        "It contains only a few sentences and should fit in a single chunk. "
        "No splitting needed here."
    )
    return Section(
        section_id="1.1",
        section_path="Chapter 1 > Section 1.1",
        heading="Introduction",
        page_start=1,
        page_end=1,
        text=text,
    )


@pytest.fixture
def large_section() -> Section:
    """A large section requiring multiple chunks (~2000+ tokens)."""
    # Generate varied sentences to create more realistic chunking
    sentences = [
        "The security controls defined in this document provide guidance. ",
        "Organizations must implement appropriate safeguards. ",
        "Controls are organized by functional areas. ",
        "Each control has specific requirements and implementation guidance. ",
        "Documentation of control implementation is essential. ",
        "Regular assessment of control effectiveness is required. ",
        "Controls must be maintained and monitored continuously. ",
        "Baseline controls provide minimum protection levels. ",
        "Enhanced controls provide additional security measures. ",
        "Tailoring of controls is permitted based on risk assessment. ",
    ] * 20  # ~2000+ tokens worth
    text = " ".join(sentences)
    return Section(
        section_id="3.1",
        section_path="Chapter 3 > Section 3.1",
        heading="Controls Overview",
        page_start=5,
        page_end=15,
        text=text,
    )


@pytest.fixture
def control_id_section() -> Section:
    """A section with NIST control ID."""
    text = (
        "Access Control (AC-2). Account Management. This control addresses the "
        "requirement to manage information system accounts. "
        "Organizations manage user accounts, group accounts, system accounts, "
        "guest/anonymous accounts, service accounts, and privilege accounts. "
        "Account management includes account creation, enablement, modification, "
        "disablement, and removal actions as defined by the organization."
    )
    return Section(
        section_id="AC-2",
        section_path="Access Control > Account Management",
        heading="AC-2 Account Management",
        page_start=10,
        page_end=12,
        text=text,
    )


@pytest.fixture
def giant_sentence_section() -> Section:
    """A section with one sentence exceeding hard cap."""
    giant_sentence = (
        "This is a single very long sentence that will definitely exceed the hard cap "
        "because it contains an enormous amount of text without natural sentence breaks "
        "and this continues on and on and on " * 20
    )
    return Section(
        section_id="2.1",
        section_path="Part 2 > Section 2.1",
        heading="Requirements",
        page_start=3,
        page_end=4,
        text=giant_sentence,
    )
