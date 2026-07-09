"""Tests for citation extraction and validation."""

from __future__ import annotations

from agentic_rag.pipeline.citations import resolve_citations
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


def make_chunk(chunk_id: str, text: str = "Test content.") -> ChunkRecord:
    """Create a test chunk."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="test",
        section_id=f"SEC{chunk_id[-1]}",
        section_ids=[f"SEC{chunk_id[-1]}"],
        section_path="Test",
        heading="Test Section",
        page_start=1,
        page_end=1,
        token_count=10,
        text=text,
    )


def make_scored(chunk: ChunkRecord, rank: int) -> ScoredChunk:
    """Create a scored chunk."""
    return ScoredChunk(chunk=chunk, score=0.9, rank=rank)


def test_multiple_citations_first_appearance_order() -> None:
    """Test that citations appear in order of first appearance."""
    chunk1 = make_chunk("c1", text="First chunk.")
    chunk2 = make_chunk("c2", text="Second chunk.")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    text = "A [1] and B [2][1]."
    result = resolve_citations(text, context)

    # Citations should be [1, 2] in order of first appearance
    assert len(result.citations) == 2
    assert result.citations[0].marker == 1
    assert result.citations[1].marker == 2
    assert result.invalid_markers == []
    assert result.text == text  # No invalid markers, text unchanged


def test_invalid_marker_stripped() -> None:
    """Test that invalid markers are removed from text."""
    chunk1 = make_chunk("c1")
    chunk2 = make_chunk("c2")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    text = "[3] claim [1]"
    result = resolve_citations(text, context)

    # Marker 3 is invalid, 1 is valid
    assert len(result.citations) == 1
    assert result.citations[0].marker == 1
    assert result.invalid_markers == [3]
    # The text should have [3] removed: "claim [1]"
    assert result.text == "claim [1]"


def test_all_invalid_markers() -> None:
    """Test when all markers are invalid."""
    chunk1 = make_chunk("c1")
    chunk2 = make_chunk("c2")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    text = "[0] and [99]"
    result = resolve_citations(text, context)

    assert result.citations == []
    assert sorted(result.invalid_markers) == [0, 99]
    # Both invalid markers removed
    assert result.text == "and"


def test_no_markers() -> None:
    """Test text without any markers."""
    chunk1 = make_chunk("c1")
    context = [make_scored(chunk1, rank=1)]

    text = "This is plain text without markers."
    result = resolve_citations(text, context)

    assert result.citations == []
    assert result.invalid_markers == []
    assert result.text == text


def test_punctuation_cleanup() -> None:
    """Test cleanup of space before punctuation artifacts."""
    chunk1 = make_chunk("c1")
    chunk2 = make_chunk("c2")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    # After removing [9], we have "Claim  ." which should become "Claim."
    text = "Claim [9] ."
    result = resolve_citations(text, context)

    # [9] is invalid (> 2)
    assert result.invalid_markers == [9]
    # After removing [9], we have "Claim  ." -> should be "Claim."
    assert result.text == "Claim."


def test_marker_zero() -> None:
    """Test that marker 0 is considered invalid."""
    chunk1 = make_chunk("c1")
    context = [make_scored(chunk1, rank=1)]

    text = "Something [0] here."
    result = resolve_citations(text, context)

    assert result.citations == []
    assert 0 in result.invalid_markers
    assert "[0]" not in result.text


def test_multiple_spaces_collapsed() -> None:
    """Test that multiple spaces are collapsed."""
    chunk1 = make_chunk("c1")
    context = [make_scored(chunk1, rank=1)]

    # After removing [5], we have "Hello   world"
    text = "Hello [5]   world"
    result = resolve_citations(text, context)

    assert 5 in result.invalid_markers
    # Multiple spaces should be collapsed to one
    assert "Hello  " not in result.text
    assert result.text == "Hello world"


def test_citation_resolves_to_chunk() -> None:
    """Test that citations resolve to the correct chunks."""
    chunk1 = make_chunk("c1", text="First content")
    chunk2 = make_chunk("c2", text="Second content")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    text = "Use [2] and [1]."
    result = resolve_citations(text, context)

    assert len(result.citations) == 2
    # First appearance order: [2] then [1]
    assert result.citations[0].marker == 2
    assert result.citations[0].chunk.chunk_id == "c2"
    assert result.citations[1].marker == 1
    assert result.citations[1].chunk.chunk_id == "c1"


def test_duplicate_markers_first_appearance() -> None:
    """Test that duplicate markers appear only once in citations."""
    chunk1 = make_chunk("c1")
    chunk2 = make_chunk("c2")

    context = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    text = "A [1] and B [1] and C [2]."
    result = resolve_citations(text, context)

    # Should have [1, 2] in first-appearance order, not duplicates
    assert len(result.citations) == 2
    assert result.citations[0].marker == 1
    assert result.citations[1].marker == 2
    assert result.invalid_markers == []


def test_space_after_punctuation_not_removed() -> None:
    """Test that we only remove space BEFORE punctuation, not after."""
    chunk1 = make_chunk("c1")
    context = [make_scored(chunk1, rank=1)]

    text = "End. [5] Start"
    result = resolve_citations(text, context)

    # After removing [5]: "End.  Start" -> "End. Start" (collapsed, no space
    # removed after period)
    assert result.text == "End. Start"


def test_complex_punctuation_cleanup() -> None:
    """Test cleanup with multiple punctuation marks."""
    chunk1 = make_chunk("c1")
    context = [make_scored(chunk1, rank=1)]

    # Multiple artifacts
    text = "Test [9] . More [8] , text [7] !"
    result = resolve_citations(text, context)

    # All markers invalid
    assert set(result.invalid_markers) == {9, 8, 7}
    # After removing all markers: "Test   .  More   ,  text   !"
    # Should clean to: "Test. More, text!"
    assert result.text == "Test. More, text!"


def test_boundary_markers() -> None:
    """Test boundary cases for marker validity (1 and max)."""
    chunk1 = make_chunk("c1")
    chunk2 = make_chunk("c2")
    chunk3 = make_chunk("c3")

    context = [
        make_scored(chunk1, rank=1),
        make_scored(chunk2, rank=2),
        make_scored(chunk3, rank=3),
    ]

    text = "First [1], last [3], invalid [4], zero [0]."
    result = resolve_citations(text, context)

    # Valid: 1, 3; Invalid: 4, 0
    assert len(result.citations) == 2
    assert {c.marker for c in result.citations} == {1, 3}
    assert set(result.invalid_markers) == {4, 0}
