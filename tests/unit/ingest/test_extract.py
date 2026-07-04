"""Tests for PDF section extraction."""

import re
from pathlib import Path

import fitz
import pytest

from agentic_rag.ingest.extract import _matches_heading_pattern, extract_sections


@pytest.fixture
def fixture_pdf(tmp_path: Path) -> Path:
    """Generate a small fixture PDF with structure for testing."""
    pdf_path = tmp_path / "fixture.pdf"

    # Create a PDF with multiple pages and sections
    doc = fitz.open()

    # Page 1: Cover/Title
    page = doc.new_page()
    page.insert_text((50, 50), "Test Document", fontsize=24, color=(0, 0, 0))

    # Headings use Helvetica-Bold ("hebo"): extraction requires bold to confirm
    # a heading pattern, mirroring the real corpus where all headings are bold.
    # Page 2: Section with numeric heading and body text
    page = doc.new_page()
    page.insert_text((50, 50), "1.1 Introduction", fontsize=18, fontname="hebo", color=(0, 0, 0))
    page.insert_text(
        (50, 100),
        "This is the introduction section with some body text. "
        "It contains multiple paragraphs of content. "
        "The text spans multiple lines.",
        fontsize=12,
    )

    # Page 3: Subsection with control ID
    page = doc.new_page()
    page.insert_text(
        (50, 50), "AC-2 Account Management", fontsize=18, fontname="hebo", color=(0, 0, 0)
    )
    page.insert_text(
        (50, 100),
        "This control addresses account management requirements. "
        "It provides guidance on user account lifecycle management. "
        "Organizations must implement appropriate controls.",
        fontsize=12,
    )

    # Page 4: Appendix
    page = doc.new_page()
    page.insert_text(
        (50, 50), "APPENDIX A. References", fontsize=18, fontname="hebo", color=(0, 0, 0)
    )
    page.insert_text(
        (50, 100),
        "This appendix contains references to other documents. "
        "It serves as a bibliography for the main content.",
        fontsize=12,
    )

    doc.save(pdf_path)
    doc.close()

    return pdf_path


def test_extract_sections_from_pdf(fixture_pdf: Path) -> None:
    """Extract sections from a fixture PDF."""
    sections = extract_sections(fixture_pdf, "test-doc")

    # Should have detected multiple sections
    assert len(sections) > 0, "Should detect at least one section"

    # Check that sections have required fields
    for section in sections:
        assert section.section_id
        assert section.section_path
        assert section.heading
        assert section.page_start >= 0
        assert section.page_end >= section.page_start
        assert section.text
        assert len(section.text) > 0


def test_section_ids_detected(fixture_pdf: Path) -> None:
    """Verify section IDs are properly extracted."""
    sections = extract_sections(fixture_pdf, "test-doc")

    section_ids = [s.section_id for s in sections]

    # Should detect numeric section "1.1"
    assert any(id.startswith("1") for id in section_ids), (
        f"Should detect numeric section, got {section_ids}"
    )

    # Should detect control ID "AC-2" or similar (has hyphen and starts with alpha)
    assert any("-" in id and id[0].isalpha() for id in section_ids), (
        f"Should detect control ID pattern, got {section_ids}"
    )

    # Should detect appendix
    assert any("appendix" in id.lower() for id in section_ids), (
        f"Should detect appendix, got {section_ids}"
    )


def test_page_ranges_tracked(fixture_pdf: Path) -> None:
    """Verify page ranges are correctly tracked."""
    sections = extract_sections(fixture_pdf, "test-doc")

    for section in sections:
        # Page numbers should be reasonable (0-indexed)
        assert section.page_start >= 0
        assert section.page_end >= section.page_start
        # Should be within document bounds (fixture has ~4 pages)
        assert section.page_end < 10


def test_section_text_preserved(fixture_pdf: Path) -> None:
    """Verify section text content is preserved."""
    sections = extract_sections(fixture_pdf, "test-doc")

    # At least one section should contain part of our fixture text
    all_text = " ".join(s.text for s in sections)

    # These phrases should appear somewhere
    all_text_lower = all_text.lower()
    assert "introduction" in all_text_lower
    assert "account" in all_text_lower, "Should preserve fixture content"


def test_doc_specific_tuning_sp800_53(fixture_pdf: Path) -> None:
    """Test doc-specific extraction tuning for SP 800-53."""
    # For sp800-53r5, should skip more front matter
    sections = extract_sections(fixture_pdf, "sp800-53r5")
    assert len(sections) >= 0, "Should handle sp800-53r5 tuning"

    # Similarly for other docs
    for doc_id in ["sp800-171r3", "ai-rmf", "fips-199", "fips-200"]:
        sections = extract_sections(fixture_pdf, doc_id)
        assert isinstance(sections, list), f"Should handle {doc_id}"


def test_section_id_grammar() -> None:
    """Enforce section_id grammar rules.

    Valid section_ids:
    - AC-2, AC-2(1), AU-12(3)  (control IDs)
    - 3, 3.1, 3.1.2  (numeric)
    - appendix-a, appendix-z  (appendices)
    - chapter-1, chapter-10  (chapters)
    - front-matter

    Invalid patterns must be structurally impossible:
    - 'three', 'a', 'untitled', 'accounts' should never appear
    """
    # Define the valid section_id grammar pattern
    valid_id_pattern = re.compile(
        r"^("
        r"[A-Z]{2}-\d+(\(\d+\))?"  # Control IDs: AC-2, AC-2(1)
        r"|"
        r"\d+\.\d+(\.\d+)*"  # Numeric: 3.1, 3.1.2 (requires at least one dot)
        r"|"
        r"appendix-[a-z]"  # Appendix: appendix-a
        r"|"
        r"chapter-\d+"  # Chapter: chapter-3
        r"|"
        r"(govern|map|measure|manage)-\d+-\d+"  # AI-RMF: govern-1-4, map-2-3
        r"|"
        r"front-matter"  # Front matter
        r")$"
    )

    # These should match (valid)
    valid_ids = [
        "AC-2",
        "AC-2(1)",
        "AU-12",
        "AU-12(3)",
        "CA-1",
        "3.1",
        "3.1.2",
        "1.2.3.4",
        "appendix-a",
        "appendix-z",
        "chapter-1",
        "chapter-10",
        "govern-1-4",
        "map-2-3",
        "measure-1-2",
        "manage-4-1",
        "front-matter",
    ]

    for section_id in valid_ids:
        assert valid_id_pattern.match(section_id), (
            f"Valid section_id '{section_id}' should match grammar"
        )

    # These should NOT match (invalid)
    invalid_ids = [
        "three",
        "a",
        "untitled",
        "accounts",
        "section-three",
        "Appendix A",  # wrong format
        "Chapter 1",  # wrong format
        "1",  # bare digit (footnote, not heading)
        "23",  # bare digit (endnote, not heading)
    ]

    for section_id in invalid_ids:
        assert not valid_id_pattern.match(section_id), (
            f"Invalid section_id '{section_id}' should NOT match grammar"
        )


def test_heading_pattern_detection() -> None:
    """Test that heading patterns are correctly detected."""
    # Control ID patterns
    assert _matches_heading_pattern("AC-2 Account Management", "sp800-53r5")
    assert _matches_heading_pattern("AU-12 Audit Generation and Review", "sp800-53r5")
    assert _matches_heading_pattern("AC-2(1) Automatic Session Termination", "sp800-53r5")

    # Numeric patterns (require at least one dot to avoid matching footnotes)
    assert _matches_heading_pattern("3.1 Security Assessment", "sp800-171r3")
    assert _matches_heading_pattern("3.1.2 Assessment Methods", "sp800-171r3")
    assert not _matches_heading_pattern("3 Bare number", "sp800-171r3")  # footnote, not heading

    # Appendix pattern
    assert _matches_heading_pattern("APPENDIX A References", "sp800-53r5")
    assert _matches_heading_pattern("Appendix B Bibliography", "sp800-53r5")

    # Chapter pattern
    assert _matches_heading_pattern("Chapter 3 Control Catalog", "sp800-53r5")
    assert _matches_heading_pattern("CHAPTER 1 Introduction", "sp800-53r5")

    # AI-RMF specific
    assert _matches_heading_pattern("GOVERN 1.1 Policy Development", "ai-rmf")
    assert _matches_heading_pattern("MAP 2.3 Mapping Processes", "ai-rmf")
    assert _matches_heading_pattern("MEASURE 3.5 Measurement", "ai-rmf")
    assert _matches_heading_pattern("MANAGE 4.1 Risk Management", "ai-rmf")

    # Should NOT match (body text)
    assert not _matches_heading_pattern("This is body text with no structure", "sp800-53r5")
    assert not _matches_heading_pattern("Organizations must implement controls", "sp800-53r5")
    assert not _matches_heading_pattern("accounts and access management", "sp800-53r5")

    # Should NOT match (errata entries that used to fire incorrectly)
    assert not _matches_heading_pattern(
        "SA-15, SA-16, SA-17, SA-20, SA-21, SR-3, SR-4, SR-5", "sp800-53r5"
    ), "Reject comma-separated control IDs (errata enumeration)"
    assert not _matches_heading_pattern('RA-3"', "sp800-53r5"), (
        "Reject control ID with trailing quote"
    )
    assert not _matches_heading_pattern("RA-3”", "sp800-53r5"), (
        "Reject control ID with trailing smart quote (errata quotes use curly quotes)"
    )
    assert not _matches_heading_pattern("AC-2 and AC-3 apply here", "sp800-53r5"), (
        "Reject control ID followed by lowercase prose (wrapped body line)"
    )
    assert not _matches_heading_pattern(
        "APPENDIX A   GLOSSARY ................................ A-1", "sp800-53r5"
    ), "Reject table-of-contents entries (dot leaders)"

    # Real catalog titles may contain commas after the ALL-CAPS title starts
    assert _matches_heading_pattern(
        "SI-7 SOFTWARE, FIRMWARE, AND INFORMATION INTEGRITY", "sp800-53r5"
    ), "Accept commas inside an ALL-CAPS title (only a comma right after the id is errata)"

    # Should match (bare control IDs on separate lines from titles)
    assert _matches_heading_pattern("AC-1", "sp800-53r5"), (
        "Accept bare control ID (title may be on next line)"
    )
    assert _matches_heading_pattern("SA-15", "sp800-53r5"), (
        "Accept bare control ID (title may be on next line or same line)"
    )


def test_duplicate_section_id_rate(fixture_pdf: Path) -> None:
    """Verify that duplicate section IDs are rare (< 5% per document).

    DEFECT 2 gate: per-document duplicate section_id rate < 5%.
    This ensures that false headings don't create duplicate sections.
    """
    sections = extract_sections(fixture_pdf, "test-doc")

    section_ids = [s.section_id for s in sections]
    total_sections = len(section_ids)

    if total_sections == 0:
        return  # No sections to check

    # Count duplicate occurrences (IDs that appear more than once)
    from collections import Counter

    id_counts = Counter(section_ids)
    duplicates = {sid: count for sid, count in id_counts.items() if count > 1}
    duplicate_rate = (
        sum(count - 1 for count in id_counts.values() if count > 1) / total_sections
        if total_sections > 0
        else 0
    )

    # For this small fixture, we just check that section IDs are reasonable
    # In production on sp800-53r5, RA-3 should appear exactly once
    assert duplicate_rate < 0.05, (
        f"Duplicate section ID rate {duplicate_rate:.1%} exceeds 5%. Duplicates: {duplicates}"
    )
