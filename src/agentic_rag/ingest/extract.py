"""Extract sections from PDF documents using pattern-based heading detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import fitz


class SectionDict(TypedDict):
    """Dictionary representation of a section being built."""

    section_id: str
    heading: str
    page_start: int
    page_end: int
    text: str
    section_path: str
    heading_stack: list[tuple[str, float]]


@dataclass(frozen=True)
class Section:
    """A logical section extracted from a PDF document."""

    section_id: str  # Canonical citation: "AC-2", "3.1.2", "appendix-a", "front-matter"
    section_path: str  # Human breadcrumb: "Chapter 3 > Section 3.1 > 3.1.2"
    heading: str  # The heading text
    page_start: int  # 0-indexed page number
    page_end: int  # 0-indexed page number (inclusive)
    text: str  # Full section text with reading order preserved


def extract_sections(pdf_path: Path, doc_id: str) -> list[Section]:
    """Extract logical sections from a PDF document.

    Uses pattern-based heading detection (control IDs, dotted numerics, APPENDIX/CHAPTER)
    anchored at line start. Font size/bold may confirm but never create headings.
    Handles doc-specific tuning via doc_id.

    Args:
        pdf_path: Path to PDF file.
        doc_id: Document identifier (e.g., "sp800-53r5") for doc-specific tuning.

    Returns:
        List of Section objects in document order.
    """
    doc = fitz.open(pdf_path)

    # Skip front matter pages (heuristic: first N pages are cover/TOC/etc)
    skip_pages = _get_skip_pages(doc_id)

    sections: list[SectionDict] = []
    current_section: SectionDict | None = None
    heading_stack: list[tuple[str, float]] = []  # Hierarchical breadcrumb of headings
    found_first_heading = False

    for page_num in range(len(doc)):
        if page_num < skip_pages:
            continue

        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:  # Not text
                continue

            for line in block["lines"]:
                # Collect all text in the line (may be multiple spans)
                line_text = ""
                max_font_size = 0.0
                is_bold = False

                for span in line["spans"]:
                    text = span["text"]
                    if text.strip():
                        line_text += text
                        max_font_size = max(max_font_size, span["size"])
                        is_bold = is_bold or bool(span["flags"] & 16)

                line_text = line_text.strip()
                if not line_text:
                    continue

                # A heading needs both: a structural pattern match AND bold font.
                # Patterns propose; bold confirms. Verified against all 5 corpus PDFs:
                # real headings are bold, while running page headers, errata rows,
                # and footnotes that mimic heading shapes are not.
                is_heading = is_bold and _matches_heading_pattern(line_text, doc_id)

                if is_heading:
                    # Save previous section if it exists
                    if current_section is not None:
                        sections.append(current_section)

                    # Detect section type and ID
                    section_id = _extract_section_id(line_text, doc_id)
                    section_path = _build_section_path(heading_stack, line_text)

                    current_section = {
                        "section_id": section_id,
                        "heading": line_text,
                        "page_start": page_num,
                        "page_end": page_num,
                        "text": "",
                        "section_path": section_path,
                        "heading_stack": heading_stack.copy(),
                    }
                    # Update hierarchy
                    _update_heading_stack(heading_stack, line_text, max_font_size)
                    found_first_heading = True
                else:
                    # Regular body text
                    if current_section is not None:
                        current_section["text"] += line_text + " "
                        current_section["page_end"] = page_num

    # Save last section
    if current_section is not None:
        sections.append(current_section)

    # Ensure front-matter section if nothing was found
    if not found_first_heading:
        # No valid headings found; create a front-matter section with all content
        sections = _create_front_matter_fallback(sections, doc, skip_pages)

    # Post-process: clean sections, strip headers/footers
    sections = _clean_sections(sections, doc, skip_pages)

    # Convert to Section dataclass
    result = [
        Section(
            section_id=s["section_id"],
            section_path=s["section_path"],
            heading=s["heading"],
            page_start=s["page_start"],
            page_end=s["page_end"],
            text=_strip_headers_footers(s["text"]),
        )
        for s in sections
    ]

    doc.close()
    return result


def _get_skip_pages(doc_id: str) -> int:
    """Return number of front-matter pages to skip by document."""
    # NIST documents typically have 5-10 pages of front matter
    skip_map = {
        "sp800-53r5": 8,
        "sp800-171r3": 6,
        "ai-rmf": 6,
        "fips-199": 2,
        "fips-200": 2,
    }
    # For test/unknown docs, skip minimal pages
    return skip_map.get(doc_id, 1)


def _matches_heading_pattern(text: str, doc_id: str) -> bool:
    r"""Check if text matches a heading pattern.

    Heading patterns (anchored at line start):
    - Control IDs: ^[A-Z]{2}-\d+(\(\d+\))? followed by end-of-line or an ALL-CAPS
      title. The boundary lookahead rejects errata rows like "SA-15, SA-16, ..."
      (comma after id) and 'RA-3”' (quote after id) while keeping real titles
      that contain commas ("SI-7 SOFTWARE, FIRMWARE, AND INFORMATION INTEGRITY").
    - Dotted numerics: ^\d+\.\d+(\.\d+)* (at least one dot; the space before the
      title is optional because some PDFs squish spans, e.g. "1.1PURPOSE...")
    - APPENDIX/CHAPTER (case-insensitive; the bold requirement in the caller
      rejects non-bold running page headers like "APPENDIX A" at size 8)
    - AI-RMF: ^(GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+ (case-insensitive)

    Lines with TOC dot leaders ("....") never match: TOC entries are bold, so
    they would otherwise survive the caller's font confirmation.
    """
    text_stripped = text.strip()

    # Table-of-contents entries use dot leaders and duplicate real heading text
    if "...." in text_stripped:
        return False

    # Control ID pattern: AC-2, AU-12(3), etc. — bare on its own line, or
    # immediately followed by an uppercase title (catalog titles are ALL CAPS).
    if re.match(r"^[A-Z]{2}-\d+(\(\d+\))?(?=$|\s+[A-Z(])", text_stripped):
        return True

    # Dotted numeric pattern: 3.1.2, 03.13.11, etc. (at least one dot to avoid
    # footnote numbers; space before title optional due to span squishing)
    if re.match(r"^\d+\.\d+(\.\d+)*\s*\S", text_stripped):
        return True

    # Appendix pattern (case-insensitive)
    if re.match(r"^appendix\s+[a-z]", text_stripped, re.IGNORECASE):
        return True

    # Chapter pattern (case-insensitive)
    if re.match(r"^chapter\s+\d+", text_stripped, re.IGNORECASE):
        return True

    # AI-RMF specific: GOVERN 1.1, MAP 2.3, etc.
    return doc_id == "ai-rmf" and bool(
        re.match(r"^(govern|map|measure|manage)\s+\d+\.\d+", text_stripped, re.IGNORECASE)
    )


def _extract_section_id(text: str, doc_id: str) -> str:
    """Extract canonical section ID from heading text.

    Enforces strict section_id grammar:
    - Control IDs: AC-2, AC-2(1)
    - Numeric: 3.1.2, 3.1
    - Appendix: appendix-a, appendix-b
    - Chapter: chapter-3
    - AI-RMF categories: govern-1-1, map-2-3, measure-1-2, manage-4-1
    - Front-matter: front-matter

    Invalid patterns (three, a, untitled, accounts) are structurally impossible.

    Args:
        text: Heading text.
        doc_id: Document identifier for doc-specific parsing.

    Returns:
        Valid section_id matching the grammar, or front-matter as fallback.
    """
    text_stripped = text.strip()

    # Control ID pattern: AC-2, AU-12(3), etc.
    control_match = re.match(r"^([A-Z]{2})-(\d+)(?:\((\d+)\))?", text_stripped)
    if control_match:
        # Format: AC-2 or AC-2(1)
        base = f"{control_match.group(1)}-{control_match.group(2)}"
        if control_match.group(3):
            base += f"({control_match.group(3)})"
        return base

    # AI-RMF pattern: GOVERN 1.4, MAP 2.3, MEASURE 1.2, MANAGE 4.1
    ai_rmf_match = re.match(
        r"^(govern|map|measure|manage)\s+(\d+)\.(\d+)", text_stripped, re.IGNORECASE
    )
    if ai_rmf_match:
        category = ai_rmf_match.group(1).lower()
        major = ai_rmf_match.group(2)
        minor = ai_rmf_match.group(3)
        return f"{category}-{major}-{minor}"

    # Dotted numeric pattern: 3.1.2, 3.1 (requires at least one dot;
    # no trailing-space requirement — some PDFs squish the title against the number)
    numeric_match = re.match(r"^(\d+\.\d+(?:\.\d+)*)", text_stripped)
    if numeric_match:
        return numeric_match.group(1)

    # Appendix pattern: appendix-a, appendix-b, etc.
    appendix_match = re.match(r"^appendix\s+([a-z])", text_stripped, re.IGNORECASE)
    if appendix_match:
        return f"appendix-{appendix_match.group(1).lower()}"

    # Chapter pattern: chapter-1, chapter-2, etc.
    chapter_match = re.match(r"^chapter\s+(\d+)", text_stripped, re.IGNORECASE)
    if chapter_match:
        return f"chapter-{chapter_match.group(1)}"

    # No valid pattern matched; shouldn't reach here if _matches_heading_pattern was true
    return "front-matter"


def _build_section_path(heading_stack: list[tuple[str, float]], current_text: str) -> str:
    """Build human-readable section path from heading hierarchy."""
    path_parts = [h[0] for h in heading_stack] + [current_text]
    return " > ".join(path_parts[-3:])  # Keep last 3 levels for readability


def _update_heading_stack(stack: list[tuple[str, float]], text: str, font_size: float) -> None:
    """Update hierarchical heading stack based on font size."""
    # Remove headings that are smaller (lower hierarchy)
    stack[:] = [(h, sz) for h, sz in stack if sz >= font_size]
    stack.append((text, font_size))


def _create_front_matter_fallback(
    sections: list[SectionDict], doc: fitz.Document, skip_pages: int
) -> list[SectionDict]:
    """Create a front-matter section if no valid headings were detected.

    Used as fallback when the document has no sections matching heading patterns.
    """
    if not sections:
        return []

    # Combine all accumulated text into front-matter
    all_text = ""
    min_page = skip_pages
    max_page = skip_pages

    for section in sections:
        all_text += section["text"] + " "
        min_page = min(min_page, section["page_start"])
        max_page = max(max_page, section["page_end"])

    if all_text.strip():
        return [
            {
                "section_id": "front-matter",
                "heading": "Front Matter",
                "page_start": min_page,
                "page_end": max_page,
                "text": all_text,
                "section_path": "Front Matter",
                "heading_stack": [],
            }
        ]

    return []


def _clean_sections(
    sections: list[SectionDict], doc: fitz.Document, skip_pages: int
) -> list[SectionDict]:
    """Clean sections: remove tiny sections, preserve substantial ones."""
    # Filter sections with substantial content (at least 50 chars)
    cleaned = [s for s in sections if len(s["text"].strip()) > 50]

    # If no sections survive filtering, create fallback front-matter
    if not cleaned and sections:
        cleaned = _create_front_matter_fallback(sections, doc, skip_pages)

    return cleaned


def _strip_headers_footers(text: str) -> str:
    """Remove common headers/footers (page numbers, running heads, etc)."""
    # Remove repeated lines that look like headers (NIST SP 800-53, REV. 5)
    lines = text.split("\n")
    filtered = []
    prev_line = ""

    for line in lines:
        line_clean = line.strip()
        # Skip lines that are exact repeats of previous (running headers)
        if (
            line_clean
            and line_clean != prev_line
            and not re.match(r"^\d+\s*$", line_clean)
            and not re.match(r"^NIST\s+SP\s+800", line_clean)
        ):
            filtered.append(line)
            prev_line = line_clean

    return "\n".join(filtered).strip()
