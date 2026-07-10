"""Pipeline contracts. Frozen surface between context building, synthesis,
citation validation, and the CLI/eval consumers.

The synthesis prompt instructs the model to begin its reply with exactly
``NO_ANSWER_SENTINEL`` when the provided context cannot answer the question.
The post-processor detects the sentinel, strips it, and sets ``Answer.refusal``;
the streaming CLI buffers the first few characters so the sentinel never
reaches the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk

NO_ANSWER_SENTINEL = "[NO_ANSWER]"


@dataclass(frozen=True, slots=True)
class SentinelScrub:
    """Result of shared sentinel post-processing (see ``scrub_sentinel``).

    ``refusal`` is True only for a leading sentinel. ``stray_sentinel`` records
    the week-5 cross-provider failure mode: a partial answer with the sentinel
    embedded mid-text or appended after it (sonnet agentic v2-q44, gemini
    vanilla v1-q18) — the sentinel is removed but the reply is NOT a refusal.
    """

    text: str
    refusal: bool
    stray_sentinel: bool


def _remove_stray_sentinels(text: str) -> str:
    """Remove every sentinel occurrence, leaving at most one space per seam."""
    pieces: list[str] = []
    i = 0
    while (j := text.find(NO_ANSWER_SENTINEL, i)) != -1:
        pieces.append(text[i:j])
        k = j + len(NO_ANSWER_SENTINEL)
        while k < len(text) and text[k] in " \t":
            k += 1
        joined = "".join(pieces)
        if joined and not joined[-1].isspace() and k < len(text) and not text[k].isspace():
            pieces.append(" ")
        i = k
    pieces.append(text[i:])
    return "".join(pieces)


def scrub_sentinel(raw: str) -> SentinelScrub:
    """Shared ``[NO_ANSWER]`` post-processing for every synthesis path.

    A reply that starts with the sentinel is a refusal: the sentinel is
    stripped and the remainder is kept as the refusal explanation. A sentinel
    anywhere else means the model answered partially and then appended a
    refusal note; every occurrence is removed (the surrounding text — usually
    an honest caveat sentence — is kept) and ``stray_sentinel`` is set.

    Streaming callers still buffer only the leading sentinel; a stray one can
    reach the live terminal but never the final ``Answer.text``.
    """
    text = raw.lstrip()
    refusal = text.startswith(NO_ANSWER_SENTINEL)
    if refusal:
        text = text[len(NO_ANSWER_SENTINEL) :].lstrip()
    stray_sentinel = NO_ANSWER_SENTINEL in text
    if stray_sentinel:
        text = _remove_stray_sentinels(text)
    return SentinelScrub(text=text.strip(), refusal=refusal, stray_sentinel=stray_sentinel)


@dataclass(frozen=True, slots=True)
class CitedChunk:
    """One resolved inline citation: the ``[n]`` marker and its chunk."""

    marker: int
    chunk: ChunkRecord


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall-clock seconds spent in one pipeline stage (retrieve/rerank/synthesize)."""

    stage: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Answer:
    """Final pipeline output.

    ``text`` carries inline ``[n]`` markers; ``citations`` resolves each valid
    marker to its chunk. Markers that referenced no provided chunk are stripped
    from ``text`` and listed in ``invalid_citations``. ``context`` is the exact
    post-rerank chunk list shown to the model (marker n == context[n-1]).
    ``usage`` sums all LLM calls (rerank + synthesis). ``refusal`` is True when
    the model correctly reported the corpus cannot answer.

    ``refusal_reason`` is a machine-readable reason set by the guardrails
    layer (``agentic_rag.guardrails.base.RefusalReason`` values: out_of_corpus,
    input_pii, input_injection, output_pii); None when guardrails are disabled.
    """

    text: str
    citations: list[CitedChunk]
    context: list[ScoredChunk]
    usage: Usage
    timings: list[StageTiming] = field(default_factory=list)
    refusal: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    refusal_reason: str | None = None
