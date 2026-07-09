"""Context builder: format and budget retrieved chunks for the synthesis prompt."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agentic_rag.retrieval.base import ScoredChunk


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """Formatted context block ready for synthesis.

    ``text`` is the numbered excerpt block passed to the model.
    ``chunks`` lists the included chunks in marker order (marker n == chunks[n-1]).
    ``token_count`` is the token count of ``text`` per the injected counter.
    """

    text: str
    chunks: list[ScoredChunk]
    token_count: int


def build_context(
    chunks: Sequence[ScoredChunk],
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> BuiltContext:
    """Build a numbered context block from chunks within a token budget.

    Excerpt format for marker n (1-based, in the given rank order):
    ``[n] {doc_id} §{section_id} — {heading} (p.{page_start})\\n{chunk text}\\n``

    Greedily adds whole excerpts in order while cumulative token count stays
    <= max_tokens. STOPS at the first excerpt that does not fit (never skips).
    At least the first excerpt is always included even if it alone exceeds
    max_tokens (an empty context is never useful).

    Args:
        chunks: Ranked list of retrieved chunks.
        max_tokens: Maximum tokens for the context block.
        count_tokens: Function to count tokens in a string.

    Returns:
        BuiltContext with formatted text, included chunks, and final token count.
    """
    if not chunks:
        return BuiltContext(text="", chunks=[], token_count=0)

    included: list[ScoredChunk] = []
    text_lines: list[str] = []
    current_tokens = 0

    for i, scored_chunk in enumerate(chunks):
        chunk = scored_chunk.chunk
        marker = i + 1  # 1-based marker number

        # Format the excerpt
        excerpt = (
            f"[{marker}] {chunk.doc_id} §{chunk.section_id} — "
            f"{chunk.heading} (p.{chunk.page_start})\n{chunk.text}\n"
        )

        # Count tokens for this excerpt
        excerpt_tokens = count_tokens(excerpt)

        # Always include the first chunk, regardless of budget
        if i == 0:
            text_lines.append(excerpt)
            included.append(scored_chunk)
            current_tokens = excerpt_tokens
        else:
            # For subsequent chunks, check if we can fit them
            total_with_this = current_tokens + excerpt_tokens
            if total_with_this <= max_tokens:
                text_lines.append(excerpt)
                included.append(scored_chunk)
                current_tokens = total_with_this
            else:
                # This chunk doesn't fit; stop here
                break

    final_text = "".join(text_lines)
    final_token_count = count_tokens(final_text)

    return BuiltContext(
        text=final_text,
        chunks=included,
        token_count=final_token_count,
    )
