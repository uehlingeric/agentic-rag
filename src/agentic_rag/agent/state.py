"""Agent loop contracts: graph state, node outputs, and the final answer shape.

Frozen surface between the graph nodes (planner, retrieve, synthesize, critic),
the LangGraph wiring in ``graph``, and the CLI/eval consumers. ``AgentState``
is the LangGraph channel schema: ``usage``, ``critiques``, and ``trace`` are
append/accumulate channels (``operator.add``), everything else is
last-write-wins. Nodes return partial updates and must therefore never re-emit
an accumulate channel value they did not produce in that step.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, TypedDict

from agentic_rag.pipeline.base import Answer
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import ScoredChunk


class PlanKind(StrEnum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"


@dataclass(frozen=True, slots=True)
class Plan:
    """Planner verdict. ``sub_queries`` is the original question verbatim for
    DIRECT plans, and 2..agent.max_sub_queries self-contained retrieval queries
    for MULTI_HOP plans, so downstream nodes never branch on the kind."""

    kind: PlanKind
    sub_queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubQueryResult:
    """Post-rerank chunks for one sub-query, before cross-query merge/dedupe."""

    query: str
    chunks: tuple[ScoredChunk, ...]


class CriticVerdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"


class IssueKind(StrEnum):
    UNCITED_CLAIM = "uncited_claim"
    UNSUPPORTED_CITATION = "unsupported_citation"
    INCOMPLETE = "incomplete"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True, slots=True)
class CriticIssue:
    """One actionable problem the critic found in a draft."""

    kind: IssueKind
    detail: str
    fix: str


@dataclass(frozen=True, slots=True)
class Critique:
    """Critic verdict for one draft. ``issues`` is empty on PASS."""

    verdict: CriticVerdict
    issues: tuple[CriticIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One node execution: wall-clock seconds plus a JSON-serializable summary
    of the node's input and output (the ``--trace`` payload)."""

    node: str
    seconds: float
    payload: dict[str, object] = field(default_factory=dict)


class AgentState(TypedDict):
    """LangGraph channel schema for one ask() invocation."""

    question: str
    plan: Plan | None
    sub_results: tuple[SubQueryResult, ...]
    context: BuiltContext | None
    draft: str
    draft_refusal: bool
    critique: Critique | None
    critiques: Annotated[tuple[Critique, ...], operator.add]
    revision_count: int
    usage: Annotated[Usage, operator.add]
    trace: Annotated[tuple[TraceEvent, ...], operator.add]


def initial_state(question: str) -> AgentState:
    """Fresh channel values for one graph invocation."""
    return AgentState(
        question=question,
        plan=None,
        sub_results=(),
        context=None,
        draft="",
        draft_refusal=False,
        critique=None,
        critiques=(),
        revision_count=0,
        usage=Usage.zero(),
        trace=(),
    )


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Final agentic output: the vanilla-shaped ``Answer`` (so CLI and eval
    consumers reuse existing rendering) plus the loop's own record.

    ``caveat`` is True when the revision cap was hit while the critic still
    said revise — the answer shipped anyway, flagged rather than rewritten.
    """

    answer: Answer
    plan: Plan
    revisions: int
    critiques: tuple[Critique, ...]
    caveat: bool
    trace: tuple[TraceEvent, ...]


def trace_to_json(events: tuple[TraceEvent, ...]) -> list[dict[str, object]]:
    """Render trace events for ``--trace`` output and trace review docs."""
    return [{"node": e.node, "seconds": round(e.seconds, 4), "payload": e.payload} for e in events]
