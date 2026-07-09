"""LLM-as-judge scoring of pipeline answers (ADR-006).

One judge call scores all three rubric dimensions — faithfulness, relevance,
citation accuracy — against the excerpts the answer actually cited, never the
reference answer. The judge must reply with a bare JSON object; a malformed
reply gets up to ``max_parse_retries`` conversational repair turns (the bad
reply plus a corrective instruction are appended) before ``JudgeParseError``
is raised. Which provider judges which answer is the caller's job: per
ADR-006 an answer is never scored by the provider that generated it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_rag.pipeline.base import CitedChunk
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, Usage

DIMENSIONS = ("faithfulness", "relevance", "citation_accuracy")

_REPAIR_INSTRUCTION = (
    "Your previous reply was not the required JSON object. Reply with ONLY the JSON "
    "object in exactly the shape given in the instructions: three dimensions, integer "
    "scores 1-5, one-sentence justifications, no code fences, no surrounding prose."
)


class JudgeParseError(Exception):
    """Judge reply was not a valid rubric JSON object after all repair attempts."""


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """One rubric dimension: integer score 1-5 plus the judge's one-line reason."""

    score: int
    justification: str


@dataclass(frozen=True, slots=True)
class JudgeScores:
    """Verdict of a single judge call, attributable to the exact judge and prompt.

    ``usage`` sums every LLM call made for this verdict, including repair turns.
    """

    faithfulness: DimensionScore
    relevance: DimensionScore
    citation_accuracy: DimensionScore
    judge_provider: str
    judge_model: str
    prompt_id: str
    usage: Usage


def format_excerpts(cited: Sequence[CitedChunk]) -> str:
    """Render cited chunks in the same numbered style the synthesis context uses."""
    if not cited:
        return "(none — the answer cited no excerpts)"
    blocks = [
        f"[{c.marker}] {c.chunk.doc_id} §{c.chunk.section_id} — {c.chunk.heading}\n{c.chunk.text}"
        for c in cited
    ]
    return "\n\n".join(blocks)


def _parse_scores(text: str) -> dict[str, DimensionScore]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in judge reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge reply is not a JSON object")
    parsed: dict[str, DimensionScore] = {}
    for dim in DIMENSIONS:
        entry = data.get(dim)
        if not isinstance(entry, dict):
            raise ValueError(f"missing dimension {dim!r}")
        score = entry.get("score")
        justification = entry.get("justification")
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"{dim}: score must be an integer, got {score!r}")
        if not 1 <= score <= 5:
            raise ValueError(f"{dim}: score {score} outside 1-5")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(f"{dim}: missing justification")
        parsed[dim] = DimensionScore(score=score, justification=justification.strip())
    return parsed


async def judge_answer(
    llm: LLMProvider,
    *,
    question: str,
    answer_text: str,
    cited: Sequence[CitedChunk],
    model: str | None = None,
    prompt_version: int | None = None,
    max_tokens: int = 768,
    max_parse_retries: int = 2,
) -> JudgeScores:
    """Score one answer on all three rubric dimensions with a single judge call."""
    prompt = load_prompt("judge", version=prompt_version)
    rendered = prompt.render(question=question, answer=answer_text, excerpts=format_excerpts(cited))
    messages: list[Message] = [Message(role=Role.USER, content=rendered)]
    usage = Usage.zero()
    last_error: ValueError | None = None
    for _ in range(max_parse_retries + 1):
        completion = await llm.complete(
            messages, model=model, max_tokens=max_tokens, temperature=0.0
        )
        usage = usage + completion.usage
        try:
            scores = _parse_scores(completion.text)
        except ValueError as exc:
            last_error = exc
            messages = [
                *messages,
                Message(role=Role.ASSISTANT, content=completion.text),
                Message(role=Role.USER, content=_REPAIR_INSTRUCTION),
            ]
            continue
        return JudgeScores(
            faithfulness=scores["faithfulness"],
            relevance=scores["relevance"],
            citation_accuracy=scores["citation_accuracy"],
            judge_provider=llm.name,
            judge_model=completion.model,
            prompt_id=prompt.id,
            usage=usage,
        )
    raise JudgeParseError(
        f"judge reply unparseable after {max_parse_retries + 1} attempts: {last_error}"
    )


def judge_provider_for(generation_provider: str, preference: Sequence[str]) -> str:
    """First judge in ``preference`` that is not the generation provider (ADR-006)."""
    for name in preference:
        if name != generation_provider:
            return name
    raise ValueError(
        f"no judge available for {generation_provider!r}: preference list {list(preference)} "
        "contains no other provider"
    )
