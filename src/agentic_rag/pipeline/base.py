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
    """

    text: str
    citations: list[CitedChunk]
    context: list[ScoredChunk]
    usage: Usage
    timings: list[StageTiming] = field(default_factory=list)
    refusal: bool = False
    invalid_citations: list[int] = field(default_factory=list)
