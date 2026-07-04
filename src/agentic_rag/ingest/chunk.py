"""Chunk extracted sections into fixed-size token windows with overlap."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from agentic_rag import tokens

if TYPE_CHECKING:
    from agentic_rag.ingest.extract import Section


@dataclass(frozen=True)
class Chunk:
    """A token-limited chunk of section text with metadata."""

    chunk_id: str
    doc_id: str
    section_id: str  # primary (first) section_id
    section_ids: list[str]  # all section_ids in merged chunk
    section_path: str
    heading: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    content_type: str  # "text" or "table"
    text: str


class _Span(NamedTuple):
    """Character span with token metadata."""

    char_start: int
    char_end: int
    token_count: int
    text: str


@dataclass
class _ConsolidatedSection:
    """A section or merged group of sections ready for chunking."""

    section_ids: list[str]  # all constituent section IDs
    section_id: str  # primary (first)
    section_path: str  # from first constituent
    heading: str  # from first constituent
    page_start: int
    page_end: int
    text: str
    token_count: int


def chunk_sections(
    doc_id: str,
    sections: list[Section],
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk sections into fixed-size overlapping token windows.

    Process: consolidate sibling sections -> chunk consolidated units -> enforce hard cap.

    Args:
        doc_id: Document identifier.
        sections: List of Section objects.
        target_tokens: Target chunk size in tokens (512).
        overlap_tokens: Overlap size between chunks (64).

    Returns:
        List of Chunk objects in document order, zero empty chunks.
    """
    # Consolidate sections: merge consecutive siblings until hitting target
    consolidated = _consolidate_sections(sections, target_tokens)

    # Chunk each consolidated unit, then enforce hard cap strictly
    chunks = []
    hard_cap = int(target_tokens * 1.5)

    for unit in consolidated:
        unit_chunks = _chunk_consolidated_unit(
            doc_id, unit, target_tokens, overlap_tokens, hard_cap
        )
        chunks.extend(unit_chunks)

    # Enforce strict hard cap: split any chunk exceeding it
    final_chunks = []
    for chunk in chunks:
        if chunk.token_count > hard_cap:
            # Split this chunk at token boundaries
            splits = _split_chunk_hard(chunk, hard_cap)
            final_chunks.extend(splits)
        else:
            final_chunks.append(chunk)

    # Assign chunk_ids last, from the per-document ordinal: this is the only
    # key guaranteed unique — window char offsets are unit-relative, so two
    # units sharing a primary section_id can produce identical-looking windows.
    return [
        replace(chunk, chunk_id=_compute_chunk_id(doc_id, chunk.section_ids, ordinal))
        for ordinal, chunk in enumerate(final_chunks)
    ]


def _consolidate_sections(
    sections: list[Section], target_tokens: int
) -> list[_ConsolidatedSection]:
    """Consolidate sections: merge consecutive siblings until hitting target_tokens.

    Rules:
    - Empty sections (<10 tokens) always merge forward.
    - Never merge across top-level units.
    - Top-level units: control families (AC-, AU-, etc.) for SP 800-53, numbered levels
      (2.x, 3.x, 4.x) for numbered docs, appendices separate.
    """
    if not sections:
        return []

    consolidated = []
    i = 0

    while i < len(sections):
        # Start a new consolidated unit
        current_unit = [sections[i]]
        current_text = sections[i].text
        current_tokens = tokens.count_tokens(current_text)
        current_ids = [sections[i].section_id]

        j = i + 1

        # Greedily merge next sections into this unit
        while j < len(sections):
            next_section = sections[j]
            next_tokens = tokens.count_tokens(next_section.text)

            # Check if next section would fit or if it's a merge-forward empty
            is_empty = next_tokens < 10
            would_fit = current_tokens + next_tokens <= target_tokens

            # Check if next section is a sibling (same top-level unit)
            is_sibling = _same_top_level_unit(current_ids[-1], next_section.section_id)

            if is_empty:
                # Always merge empty sections forward
                current_unit.append(next_section)
                current_text += " " + next_section.text
                current_tokens += next_tokens
                current_ids.append(next_section.section_id)
                j += 1
            elif is_sibling and would_fit:
                # Merge non-empty sibling that fits
                current_unit.append(next_section)
                current_text += " " + next_section.text
                current_tokens += next_tokens
                current_ids.append(next_section.section_id)
                j += 1
            else:
                # Stop: next section doesn't fit or is not a sibling
                break

        # Create consolidated unit
        first = sections[i]
        last = sections[j - 1]

        consolidated.append(
            _ConsolidatedSection(
                section_ids=current_ids,
                section_id=current_ids[0],
                section_path=first.section_path,
                heading=first.heading,
                page_start=first.page_start,
                page_end=last.page_end,
                text=current_text,
                token_count=current_tokens,
            )
        )

        i = j

    return consolidated


def _same_top_level_unit(section_id_1: str, section_id_2: str) -> bool:
    """Check if two section_ids belong to the same top-level unit.

    Top-level units (following design comment in consolidation section, line ~107):
    - Control families: AC-2, AC-3, AC-4 are same family (top-level unit AC), can merge
    - AC-x never merges with AU-x (different families)
    - Numbered: 3.1 and 3.2 same (top level 3); 3.x and 4.x differ
    - Appendices: never merge with body; appendices can merge with each other
    - More aggressive: single-digit sections (1, 2, 3) and unnumbered can merge with each
      other within reasonable scope (avoid huge merges by checking token count in caller)
    """
    s1_upper = section_id_1.upper()
    s2_upper = section_id_2.upper()

    # Control ID pattern (e.g., AC-2, AU-1) - family-level merging
    control_match_1 = re.match(r"^([A-Z]{2})-(\d+)(?:\(\d+\))?$", s1_upper)
    control_match_2 = re.match(r"^([A-Z]{2})-(\d+)(?:\(\d+\))?$", s2_upper)

    if control_match_1 and control_match_2:
        family_1 = control_match_1.group(1)
        family_2 = control_match_2.group(1)
        # Same family can merge (AC-2 with AC-3, but AC-x never with AU-x)
        return family_1 == family_2

    # Numbered section pattern (3.1.2, 3.1, 3, etc.) - only check top-level number
    numeric_match_1 = re.match(r"^(\d+)", s1_upper)
    numeric_match_2 = re.match(r"^(\d+)", s2_upper)

    if numeric_match_1 and numeric_match_2:
        # Same if both are purely numeric at top level (not mixed with text)
        top_level_1 = numeric_match_1.group(1)
        top_level_2 = numeric_match_2.group(1)
        # Check if the full ID is numeric-only or numeric.numeric pattern
        is_numeric_only_1 = bool(re.match(r"^\d+(\.\d+)*$", s1_upper))
        is_numeric_only_2 = bool(re.match(r"^\d+(\.\d+)*$", s2_upper))
        return is_numeric_only_1 and is_numeric_only_2 and top_level_1 == top_level_2

    # Appendix pattern (e.g., appendix-a, appendix-b)
    is_appendix_1 = "appendix" in s1_upper.lower()
    is_appendix_2 = "appendix" in s2_upper.lower()

    if is_appendix_1 and is_appendix_2:
        return True

    if is_appendix_1 or is_appendix_2:
        # One is appendix, one is not: don't merge
        return False

    # Single-digit sections (1, 2, 3) or "untitled": can be siblings for aggressive merging
    # This helps with sections that are poorly extracted or minimal content
    is_single_digit_1 = bool(re.match(r"^\d+$", s1_upper))
    is_single_digit_2 = bool(re.match(r"^\d+$", s2_upper))
    is_untitled_1 = "untitled" in s1_upper.lower()
    is_untitled_2 = "untitled" in s2_upper.lower()

    # Merge single-digit/untitled sections (scope enforced by token target)
    return (is_single_digit_1 or is_untitled_1) and (is_single_digit_2 or is_untitled_2)


def _chunk_consolidated_unit(
    doc_id: str,
    unit: _ConsolidatedSection,
    target_tokens: int,
    overlap_tokens: int,
    hard_cap: int,
) -> list[Chunk]:
    """Chunk a consolidated unit, respecting sentence boundaries."""
    section_text = unit.text
    section_tokens = unit.token_count

    # Small unit: single chunk
    if section_tokens <= target_tokens:
        char_start = 0
        char_end = len(section_text)
        return [
            Chunk(
                chunk_id="",  # assigned from the document ordinal in chunk_sections
                doc_id=doc_id,
                section_id=unit.section_id,
                section_ids=unit.section_ids,
                section_path=unit.section_path,
                heading=unit.heading,
                page_start=unit.page_start,
                page_end=unit.page_end,
                char_start=char_start,
                char_end=char_end,
                token_count=section_tokens,
                content_type=_detect_content_type(section_text),
                text=section_text,
            )
        ]

    # Large unit: split at sentence boundaries
    sentences = _split_sentences(section_text)
    spans = _span_sentences(sentences, section_text)

    chunks = []
    i = 0
    while i < len(spans):
        # Build chunk starting at span i, aiming for target_tokens
        chunk_spans = [spans[i]]
        chunk_tokens = spans[i].token_count

        j = i + 1
        while j < len(spans) and chunk_tokens + spans[j].token_count <= target_tokens:
            chunk_tokens += spans[j].token_count
            chunk_spans.append(spans[j])
            j += 1

        # If single span exceeds hard cap, split it
        if len(chunk_spans) == 1 and chunk_tokens > hard_cap:
            chunk_spans = _split_span_hard(spans[i], target_tokens, hard_cap)

        # Create chunk from combined spans
        char_start = chunk_spans[0].char_start
        char_end = chunk_spans[-1].char_end
        chunk_text = section_text[char_start:char_end]
        chunk_token_count = tokens.count_tokens(chunk_text)

        chunks.append(
            Chunk(
                chunk_id="",  # assigned from the document ordinal in chunk_sections
                doc_id=doc_id,
                section_id=unit.section_id,
                section_ids=unit.section_ids,
                section_path=unit.section_path,
                heading=unit.heading,
                page_start=unit.page_start,
                page_end=unit.page_end,
                char_start=char_start,
                char_end=char_end,
                token_count=chunk_token_count,
                content_type=_detect_content_type(chunk_text),
                text=chunk_text,
            )
        )

        # Find next start position for overlap
        next_start_idx = j
        overlap_count = 0
        while next_start_idx > i + 1:
            prev_idx = next_start_idx - 1
            new_count = overlap_count + spans[prev_idx].token_count
            if new_count > overlap_tokens * 2:
                break
            next_start_idx = prev_idx
            overlap_count = new_count

        i = next_start_idx
        if i >= len(spans):
            break

    # Re-pack stranded small windows: greedy packing leaves a fragment whenever
    # the next span would overflow the target (and at unit tails).
    return _repack_small_windows(chunks, section_text, overlap_tokens * 2, hard_cap)


def _repack_small_windows(
    chunks: list[Chunk], section_text: str, min_tokens: int, hard_cap: int
) -> list[Chunk]:
    """Merge windows below min_tokens into a neighbor while staying ≤ hard_cap.

    Windows within a unit are contiguous-or-overlapping char ranges over
    section_text, so a merge is just the covering range, re-tokenized.
    """
    result = list(chunks)
    i = 0
    while len(result) > 1 and i < len(result):
        small = result[i]
        if small.token_count >= min_tokens:
            i += 1
            continue

        merged_at = None
        for neighbor_idx in (i - 1, i + 1):  # prefer merging into the previous window
            if not 0 <= neighbor_idx < len(result):
                continue
            left, right = sorted((result[neighbor_idx], small), key=lambda c: c.char_start)
            merged_text = section_text[left.char_start : right.char_end]
            merged_tokens = tokens.count_tokens(merged_text)
            if merged_tokens > hard_cap:
                continue
            merged_at = min(i, neighbor_idx)
            result[merged_at : merged_at + 2] = [
                replace(
                    result[merged_at],
                    char_start=left.char_start,
                    char_end=right.char_end,
                    token_count=merged_tokens,
                    content_type=_detect_content_type(merged_text),
                    text=merged_text,
                )
            ]
            break

        # Re-examine the merged window (it may still be small); if no neighbor
        # could absorb this one without breaching the cap, leave it and move on.
        i = merged_at if merged_at is not None else i + 1

    return result


def _split_chunk_hard(chunk: Chunk, hard_cap: int) -> list[Chunk]:
    """Split a chunk that exceeds hard_cap at token boundaries.

    Produces multiple chunks, each ≤ hard_cap tokens. Pieces are evenly sized
    (ceil-divided) rather than cap-sized-plus-remainder, so no tiny tail piece.
    """
    text = chunk.text
    encoded = tokens.encode(text)
    n_pieces = math.ceil(len(encoded) / hard_cap)
    piece_size = math.ceil(len(encoded) / n_pieces)

    result = []
    i = 0
    while i < len(encoded):
        end_idx = min(i + piece_size, len(encoded))
        chunk_tokens = encoded[i:end_idx]
        chunk_text = tokens.decode(chunk_tokens)

        # Compute char offset by decoding prefix
        char_offset = len(tokens.decode(encoded[:i])) if i > 0 else 0

        char_start = chunk.char_start + char_offset
        char_end = char_start + len(chunk_text)

        result.append(
            Chunk(
                chunk_id="",  # assigned from the document ordinal in chunk_sections
                doc_id=chunk.doc_id,
                section_id=chunk.section_id,
                section_ids=chunk.section_ids,
                section_path=chunk.section_path,
                heading=chunk.heading,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                char_start=char_start,
                char_end=char_end,
                token_count=len(chunk_tokens),
                content_type=chunk.content_type,
                text=chunk_text,
            )
        )

        i = end_idx

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at natural boundaries (., !, ?)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _span_sentences(sentences: list[str], full_text: str) -> list[_Span]:
    """Map sentences back to character positions in full text."""
    spans = []
    pos = 0

    for sent in sentences:
        start = full_text.find(sent, pos)
        if start == -1:
            continue
        end = start + len(sent)
        tok_count = tokens.count_tokens(sent)
        spans.append(_Span(start, end, tok_count, sent))
        pos = end

    return spans


def _split_span_hard(span: _Span, target_tokens: int, hard_cap: int) -> list[_Span]:
    """Split a single span that exceeds hard cap at token boundaries.

    Pieces are evenly sized (ceil-divided around target_tokens) so the last
    piece is never a tiny remainder.
    """
    text = span.text
    encoded = tokens.encode(text)
    n_pieces = math.ceil(len(encoded) / target_tokens)
    piece_size = math.ceil(len(encoded) / n_pieces)

    splits = []
    i = 0
    while i < len(encoded):
        end_idx = min(i + piece_size, len(encoded))
        chunk_tokens = encoded[i:end_idx]
        chunk_text = tokens.decode(chunk_tokens)

        char_offset = len(tokens.decode(encoded[:i])) if i > 0 else 0

        splits.append(
            _Span(
                span.char_start + char_offset,
                span.char_start + char_offset + len(chunk_text),
                len(chunk_tokens),
                chunk_text,
            )
        )
        i = end_idx

    return splits


def _detect_content_type(text: str) -> str:
    """Detect if text is a table (heuristic: 3+ consecutive lines with aligned columns)."""
    lines = text.split("\n")
    if len(lines) < 3:
        return "text"

    table_lines = 0
    for line in lines:
        if not line.strip():
            continue

        space_runs = len(re.findall(r"[\s]{2,}|\t", line))
        if space_runs >= 1 and len(line.split()) >= 2:
            table_lines += 1
        else:
            table_lines = 0

        if table_lines >= 3:
            return "table"

    return "text"


def _compute_chunk_id(doc_id: str, section_ids: list[str], ordinal: int) -> str:
    """Compute deterministic chunk ID from document position.

    The per-document ordinal is the uniqueness key: char offsets are relative
    to their consolidated unit, so they can repeat across units. The primary
    section_id is included for debuggability of the hash input, not uniqueness.

    Returns:
        16-character hex hash for use as chunk_id.
    """
    primary_id = section_ids[0] if section_ids else "unknown"
    key = f"{doc_id}:{primary_id}:{ordinal:04d}"
    hash_digest = hashlib.sha1(key.encode()).hexdigest()
    return hash_digest[:16]
