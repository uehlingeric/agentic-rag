"""Citation validation: extract and validate inline markers from answer text."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_rag.pipeline.base import CitedChunk
from agentic_rag.retrieval.base import ScoredChunk


@dataclass(frozen=True, slots=True)
class CitationResult:
    """Result of citation extraction and validation.

    ``text`` has invalid markers stripped and minor whitespace cleanup applied.
    ``citations`` lists unique valid markers in order of first appearance.
    ``invalid_markers`` lists unique invalid marker numbers in ascending order.
    """

    text: str
    citations: list[CitedChunk]
    invalid_markers: list[int]


def resolve_citations(text: str, context_chunks: Sequence[ScoredChunk]) -> CitationResult:
    """Extract and validate inline citations from answer text.

    Markers match regex ``\\[(\\d+)\\]``. Marker n is valid iff
    1 <= n <= len(context_chunks); resolves to context_chunks[n-1].chunk.

    Removes invalid markers from text and cleans up artifacts:
    - Collapses runs of spaces (e.g., "  " -> " ")
    - Removes space before punctuation (e.g., " ." -> ".")

    Citations and invalid_markers preserve first-appearance order and
    ascending order respectively.

    Args:
        text: Answer text with inline ``[n]`` markers.
        context_chunks: Chunks corresponding to markers 1..N.

    Returns:
        CitationResult with cleaned text, valid citations, and invalid markers.
    """
    # Extract all markers from the text
    marker_pattern = re.compile(r"\[(\d+)\]")
    markers = [(int(m.group(1)), m.start(), m.end()) for m in marker_pattern.finditer(text)]

    valid_markers_seen: set[int] = set()
    citations: list[CitedChunk] = []
    invalid_markers_set: set[int] = set()

    # Track which markers are valid and collect in order of appearance
    for marker_num, _, _ in markers:
        is_valid = 1 <= marker_num <= len(context_chunks)

        if is_valid:
            if marker_num not in valid_markers_seen:
                valid_markers_seen.add(marker_num)
                citations.append(
                    CitedChunk(
                        marker=marker_num,
                        chunk=context_chunks[marker_num - 1].chunk,
                    )
                )
        else:
            invalid_markers_set.add(marker_num)

    # Remove invalid markers from text
    cleaned_text = text
    for marker_num, start, end in reversed(markers):
        if not (1 <= marker_num <= len(context_chunks)):
            cleaned_text = cleaned_text[:start] + cleaned_text[end:]

    # Clean up whitespace artifacts
    # Collapse multiple spaces into one
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)
    # Remove space before punctuation
    cleaned_text = re.sub(r" ([.!?,;:])", r"\1", cleaned_text)
    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()

    invalid_markers = sorted(invalid_markers_set)

    return CitationResult(
        text=cleaned_text,
        citations=citations,
        invalid_markers=invalid_markers,
    )
