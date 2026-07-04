"""Tests for chunking logic."""

from agentic_rag import tokens
from agentic_rag.ingest.chunk import chunk_sections


class TestChunkSmallSection:
    """Test that small sections produce single chunks."""

    def test_single_chunk_for_fitting_section(self, small_section):
        """Section fitting in target_tokens produces one chunk."""
        chunks = chunk_sections("test-doc", [small_section], target_tokens=512)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.doc_id == "test-doc"
        assert chunk.section_id == small_section.section_id
        assert chunk.text == small_section.text
        assert chunk.page_start == small_section.page_start
        assert chunk.page_end == small_section.page_end
        assert chunk.token_count == tokens.count_tokens(small_section.text)
        assert 0 < chunk.token_count <= 512


class TestChunkLargeSection:
    """Test splitting of large sections with overlap."""

    def test_large_section_splits_into_multiple(self, large_section):
        """Large section splits into multiple chunks."""
        chunks = chunk_sections("test-doc", [large_section], target_tokens=512, overlap_tokens=64)

        assert len(chunks) > 1, "Large section should split into multiple chunks"

        # Check overlap: each chunk except last should have 0 < token overlap with next
        for i in range(len(chunks) - 1):
            current_end = chunks[i].char_end
            next_start = chunks[i + 1].char_start

            # Chunks should overlap (next starts before current ends)
            assert next_start < current_end, "Consecutive chunks should have character overlap"

            # Verify actual token overlap is reasonable (allow some overage due to span granularity)
            overlap_text = large_section.text[next_start:current_end]
            overlap_tokens = tokens.count_tokens(overlap_text)
            assert 0 < overlap_tokens <= int(2 * 64 * 1.2), (
                f"Overlap tokens {overlap_tokens} out of range [0, ~154]"
            )

    def test_no_chunk_exceeds_hard_cap(self, large_section, giant_sentence_section):
        """No chunk should exceed hard cap (target * 1.5)."""
        all_sections = [large_section, giant_sentence_section]
        chunks = chunk_sections("test-doc", all_sections, target_tokens=512, overlap_tokens=64)

        hard_cap = int(512 * 1.5)
        for chunk in chunks:
            assert chunk.token_count <= hard_cap, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens, cap is {hard_cap}"
            )

    def test_chunks_within_or_near_target(self, large_section):
        """Most chunks should be close to target size."""
        chunks = chunk_sections("test-doc", [large_section], target_tokens=512, overlap_tokens=64)

        # With overlapping chunks, we should have reasonable chunk counts
        # At least some chunks should be substantial (>300 tokens)
        substantial = [c for c in chunks if c.token_count > 300]
        assert len(substantial) >= 1, "Should have at least one substantial chunk"


class TestChunkSentenceBoundaries:
    """Test that chunks respect sentence boundaries when possible."""

    def test_chunks_end_with_sentence_punctuation(self, large_section):
        """Chunks should end at sentence boundaries (., !, ?) when not hard-split."""
        chunks = chunk_sections("test-doc", [large_section], target_tokens=512, overlap_tokens=64)

        for chunk in chunks:
            text = chunk.text.strip()
            # Should end with sentence-final punctuation unless hard-split
            assert text[-1] in ".!?", f"Chunk should end with punctuation: {text[-30:]}"


class TestChunkDeterminism:
    """Test that chunk IDs are deterministic."""

    def test_same_input_produces_same_chunk_ids(self, small_section, large_section):
        """Running chunking twice should produce identical chunk IDs."""
        sections = [small_section, large_section]

        chunks1 = chunk_sections("test-doc", sections, target_tokens=512, overlap_tokens=64)
        chunks2 = chunk_sections("test-doc", sections, target_tokens=512, overlap_tokens=64)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1.chunk_id == c2.chunk_id, "Chunk IDs should be deterministic"


class TestChunkCharOffsets:
    """Test that char offsets correctly map back to section text."""

    def test_char_offsets_slice_original_text(self, small_section, large_section):
        """chunk.text should equal section.text[char_start:char_end]."""
        sections = [small_section, large_section]
        chunks = chunk_sections("test-doc", sections, target_tokens=512, overlap_tokens=64)

        # Map chunks back to sections for validation
        for chunk in chunks:
            # Find the section this chunk came from
            section = next((s for s in sections if s.section_id == chunk.section_id), None)
            assert section is not None

            # Verify char offsets
            assert 0 <= chunk.char_start < len(section.text)
            assert 0 < chunk.char_end <= len(section.text)
            assert chunk.char_start < chunk.char_end

            # Verify text matches
            sliced_text = section.text[chunk.char_start : chunk.char_end]
            assert chunk.text == sliced_text, "Chunk text should match section slice"


class TestChunkCrossSection:
    """Test that chunks never cross section boundaries."""

    def test_chunks_within_section_boundaries(
        self, small_section, large_section, control_id_section
    ):
        """Each chunk belongs entirely to one section."""
        sections = [small_section, large_section, control_id_section]
        chunks = chunk_sections("test-doc", sections, target_tokens=512, overlap_tokens=64)

        for chunk in chunks:
            # Find parent section
            section = next(s for s in sections if s.section_id == chunk.section_id)
            # Verify chunk text is within section
            assert 0 <= chunk.char_start < len(section.text)
            assert chunk.char_end <= len(section.text)


class TestChunkContentType:
    """Test content type detection (text vs table)."""

    def test_regular_text_marked_as_text(self, small_section):
        """Regular prose should be marked as text."""
        chunks = chunk_sections("test-doc", [small_section])
        assert all(c.content_type == "text" for c in chunks)

    def test_table_content_detected(self):
        """Sections with table-like structure should be marked as table."""
        from agentic_rag.ingest.extract import Section

        # Create a section that looks like a table (multiple spaces/tabs)
        table_text = (
            "Control ID    Control Name        Description\n"
            "AC-2          Account Management  This control addresses...\n"
            "AU-1          Audit and           Provides audit and...\n"
            "CA-1          Security Assessment Provides guidance..."
        )

        section = Section(
            section_id="table",
            section_path="Tables",
            heading="Control Summary Table",
            page_start=100,
            page_end=101,
            text=table_text,
        )

        chunks = chunk_sections("test-doc", [section])
        # Should detect table structure
        assert any(c.content_type == "table" for c in chunks)


class TestChunkIdUniqueness:
    """Test that chunk IDs are unique across all chunks (DEFECT 1 fix)."""

    def test_chunk_ids_unique_within_corpus(self, small_section, large_section, control_id_section):
        """Every chunk should have a unique chunk_id, even from same section."""
        sections = [small_section, large_section, control_id_section]
        chunks = chunk_sections("test-doc", sections, target_tokens=512, overlap_tokens=64)

        chunk_ids = [c.chunk_id for c in chunks]
        unique_ids = set(chunk_ids)

        assert len(chunk_ids) == len(unique_ids), (
            f"Found {len(chunk_ids)} chunks but only {len(unique_ids)} unique IDs. "
            "Multiple chunks should not share the same ID."
        )

        # Verify no duplicates (this is the critical assertion)
        duplicates = [cid for cid in set(chunk_ids) if chunk_ids.count(cid) > 1]
        assert not duplicates, f"Found duplicate chunk_ids: {duplicates}"
