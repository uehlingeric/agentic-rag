"""Context builder: format and budget retrieved chunks for the synthesis prompt.

Excerpt format uses XML-style delimiters to establish retrieved-content injection
defense: chunks are data, not instructions. The synthesis prompt instructs the
model to ignore any directives inside <excerpt> tags, and this builder guarantees
that excerpt boundaries (opening and closing tags) cannot be forged by chunk
content. Poisoned chunks cannot close the delimiter or create fake excerpts.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agentic_rag.retrieval.base import ScoredChunk


def _neutralize_tags(text: str) -> str:
    """Escape <excerpt and </excerpt markers in chunk text.

    Replaces any case-insensitive occurrence with HTML entity escape
    of the angle bracket only, preserving original spelling.

    Args:
        text: Chunk text that may contain poisoned excerpt-like markers.

    Returns:
        Text with <excerpt and </excerpt escaped to &lt;excerpt and &lt;/excerpt.
    """
    return re.sub(r"(?i)<(/?)(excerpt)", r"&lt;\1\2", text)


def _attr(value: str) -> str:
    """Sanitize attribute values: escape <excerpt markers and quotes.

    Replaces double quotes with single quotes (prevents breaking out of
    attributes) and neutralizes any <excerpt markers.

    Args:
        value: Attribute value (e.g., heading text).

    Returns:
        Sanitized value safe to embed in XML attributes.
    """
    sanitized = value.replace('"', "'")
    return _neutralize_tags(sanitized)


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
    ``<excerpt id={n} source="{doc_id} §{section_id} — {heading}
    (p.{page_start})">\\n{sanitized_text}\\n</excerpt>\\n``

    Text is sanitized to prevent injection: <excerpt and </excerpt markers
    in chunk text are escaped to &lt;excerpt and &lt;/excerpt. Attribute
    values (doc_id, section_id, heading) are also sanitized: quotes are
    replaced with single quotes and <excerpt markers are escaped.

    Greedily adds whole excerpts in order while cumulative token count stays
    <= max_tokens. STOPS at the first excerpt that does not fit (never skips).
    At least the first excerpt is always included even if it alone exceeds
    max_tokens (an empty context is never useful).

    Token counting includes the full formatted excerpt tags.

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

        # Sanitize chunk text to prevent injection attacks
        sanitized_text = _neutralize_tags(chunk.text)

        # Build the source attribute with sanitized values
        source = (
            f"{_attr(chunk.doc_id)} §{_attr(chunk.section_id)} — "
            f"{_attr(chunk.heading)} (p.{chunk.page_start})"
        )

        # Format the excerpt with XML-style delimiters
        excerpt = f'<excerpt id={marker} source="{source}">\n{sanitized_text}\n</excerpt>\n'

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
