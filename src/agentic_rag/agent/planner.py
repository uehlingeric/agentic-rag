"""Query planner: decide whether to decompose a question into sub-queries (ADR-007).

One LLM call classifies the question as DIRECT (single retrieval pass) or
MULTI_HOP (decomposed into 2..N independent sub-queries). MULTI_HOP decompositions
that degenerate (fewer than 2 valid sub-queries after filtering, or invalid JSON)
fall back to DIRECT silently — the loop degrades to vanilla behavior instead of
crashing. A broken planner is safe and cheap; synthesis and critic catch
problems downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentic_rag.agent.state import Plan, PlanKind
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, Usage


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Planner call outcome. ``fallback`` is True when the model reply could not
    be parsed into a valid plan and we defaulted to a DIRECT pass-through —
    the loop degrades to vanilla behavior instead of crashing."""

    plan: Plan
    usage: Usage
    prompt_id: str
    fallback: bool
    raw: str


async def plan_query(
    llm: LLMProvider,
    question: str,
    *,
    model: str | None = None,
    max_sub_queries: int = 4,
    max_tokens: int = 512,
    prompt_version: int | None = None,
) -> PlanResult:
    """Classify question as DIRECT or decompose into MULTI_HOP sub-queries.

    Args:
        llm: LLM provider for the planner call.
        question: The user's question to classify.
        model: Model override; None uses provider default.
        max_sub_queries: Maximum number of sub-queries to keep (2..max).
        max_tokens: Max tokens for the model response.
        prompt_version: Prompt version pin; None uses latest.

    Returns:
        PlanResult with the plan, usage, and parse status. On any parse failure,
        falls back to Plan(DIRECT, (question,)) with fallback=True.
    """
    prompt = load_prompt("planner", version=prompt_version)
    rendered = prompt.render(question=question)
    messages = [Message(role=Role.USER, content=rendered)]

    completion = await llm.complete(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    plan, fallback = _parse_plan(completion.text, question, max_sub_queries)

    return PlanResult(
        plan=plan,
        usage=completion.usage,
        prompt_id=prompt.id,
        fallback=fallback,
        raw=completion.text,
    )


def _parse_plan(text: str, question: str, max_sub_queries: int) -> tuple[Plan, bool]:
    """Parse planner JSON reply into a Plan.

    Args:
        text: Verbatim LLM reply.
        question: Original question (fallback sub-query).
        max_sub_queries: Maximum sub-queries to keep.

    Returns:
        Tuple of (Plan, fallback_flag). Plan is either DIRECT or MULTI_HOP;
        fallback_flag is True only when parsing failed or decomposition
        degenerated (multi_hop with <2 valid sub-queries after filtering).
    """
    # Extract JSON using find/rfind like judge._parse_scores
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True

    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True

    if not isinstance(data, dict):
        return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True

    classification = data.get("classification")

    # Handle direct classification
    if classification == "direct":
        return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), False

    # Handle multi_hop classification
    if classification == "multi_hop":
        sub_queries_raw = data.get("sub_queries")
        if not isinstance(sub_queries_raw, list):
            return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True

        # Filter: strip, drop empty/whitespace-only, drop non-strings
        filtered = []
        for entry in sub_queries_raw:
            if isinstance(entry, str):
                stripped = entry.strip()
                if stripped:
                    filtered.append(stripped)

        # If fewer than 2 remain, decomposition violated contract
        if len(filtered) < 2:
            return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True

        # Keep first max_sub_queries (not a fallback if we're truncating)
        kept = filtered[:max_sub_queries]
        return (
            Plan(kind=PlanKind.MULTI_HOP, sub_queries=tuple(kept)),
            False,
        )

    # Unknown classification
    return Plan(kind=PlanKind.DIRECT, sub_queries=(question,)), True
