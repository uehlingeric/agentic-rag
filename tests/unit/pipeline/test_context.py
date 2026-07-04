"""Tests for context building and budgeting."""

from __future__ import annotations

from agentic_rag.pipeline.context import build_context
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    heading: str = "AC-2 ACCOUNT MANAGEMENT",
    text: str = "The organization manages system accounts.",
    token_count: int = 10,
) -> ChunkRecord:
    """Create a test chunk."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=heading,
        heading=heading,
        page_start=1,
        page_end=2,
        token_count=token_count,
        text=text,
    )


def make_scored(chunk: ChunkRecord, rank: int = 1, score: float = 0.9) -> ScoredChunk:
    """Create a scored chunk."""
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def test_formatting_two_chunks() -> None:
    """Test that two chunks are formatted with correct markers and structure."""
    chunk1 = make_chunk("c1", section_id="AC-2", heading="AC-2 ACCOUNT", text="Account text.")
    chunk2 = make_chunk("c2", section_id="AU-2", heading="AU-2 AUDIT", text="Audit text.")

    scored1 = make_scored(chunk1, rank=1)
    scored2 = make_scored(chunk2, rank=2)

    def count_chars(s: str) -> int:
        return len(s)

    result = build_context([scored1, scored2], max_tokens=1000, count_tokens=count_chars)

    # Check that both chunks are included
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk.chunk_id == "c1"
    assert result.chunks[1].chunk.chunk_id == "c2"

    # Check the formatted text
    expected = (
        "[1] sp800-53r5 §AC-2 — AC-2 ACCOUNT (p.1)\n"
        "Account text.\n"
        "[2] sp800-53r5 §AU-2 — AU-2 AUDIT (p.1)\n"
        "Audit text.\n"
    )
    assert result.text == expected
    assert result.token_count == len(expected)


def test_budget_stops_not_skips() -> None:
    """Test that chunks are added greedily and stops when next doesn't fit."""
    # Create chunks with specific sizes
    chunk1 = make_chunk("c1", text="A" * 10)  # 10 chars
    chunk2 = make_chunk("c2", text="B" * 20)  # 20 chars
    chunk3 = make_chunk("c3", text="C" * 100)  # 100 chars - too big to fit

    scored = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2), make_scored(chunk3, rank=3)]

    def count_chars(s: str) -> int:
        return len(s)

    # Budget allows first and second but not third
    result = build_context(scored, max_tokens=150, count_tokens=count_chars)

    # Should include chunks 1 and 2, but not 3
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk.chunk_id == "c1"
    assert result.chunks[1].chunk.chunk_id == "c2"
    assert result.token_count == len(result.text)


def test_oversized_first_chunk_included() -> None:
    """Test that the first chunk is always included, even if oversized."""
    chunk1 = make_chunk("c1", text="X" * 200)  # Very large first chunk

    scored = [make_scored(chunk1, rank=1)]

    def count_chars(s: str) -> int:
        return len(s)

    # Budget is small but first chunk should still be included
    result = build_context(scored, max_tokens=50, count_tokens=count_chars)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk.chunk_id == "c1"
    assert result.token_count > 50  # Exceeds budget but is included


def test_empty_input() -> None:
    """Test that empty chunk list produces empty context."""

    def count_chars(s: str) -> int:
        return len(s)

    result = build_context([], max_tokens=1000, count_tokens=count_chars)

    assert result.text == ""
    assert result.chunks == []
    assert result.token_count == 0


def test_single_chunk_under_budget() -> None:
    """Test single chunk that fits within budget."""
    chunk = make_chunk("c1", section_id="XX", heading="Test", text="Content")
    scored = [make_scored(chunk, rank=1)]

    def count_chars(s: str) -> int:
        return len(s)

    result = build_context(scored, max_tokens=1000, count_tokens=count_chars)

    assert len(result.chunks) == 1
    assert "[1]" in result.text
    assert "Content" in result.text
    assert result.token_count == len(result.text)


def test_exact_budget_boundary() -> None:
    """Test chunk that exactly fits within budget."""
    chunk1 = make_chunk("c1", text="A" * 50)
    chunk2 = make_chunk("c2", text="B" * 30)

    scored = [make_scored(chunk1, rank=1), make_scored(chunk2, rank=2)]

    def count_chars(s: str) -> int:
        return len(s)

    # Build context for chunk1 and measure; then use that as budget
    ctx1 = build_context([scored[0]], max_tokens=1000, count_tokens=count_chars)
    chunk1_text_size = len(ctx1.text)

    # Now allow exactly enough for both
    result = build_context(scored, max_tokens=chunk1_text_size + 200, count_tokens=count_chars)

    assert len(result.chunks) == 2
