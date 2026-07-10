"""Tests for the agent loop graph orchestration."""

from __future__ import annotations

import pytest

from agentic_rag.agent.graph import build_graph
from agentic_rag.agent.state import (
    AgentState,
    CriticVerdict,
    Critique,
    TraceEvent,
    initial_state,
)
from agentic_rag.providers.base import Usage


class GraphTestHarness:
    """Tracks node executions and call counts for assertions."""

    def __init__(self) -> None:
        self.planner_count = 0
        self.retrieve_count = 0
        self.synthesize_count = 0
        self.critic_count = 0


@pytest.mark.asyncio
async def test_critic_passes_on_first_draft() -> None:
    """Critic passes on first draft: synthesize runs once, revision_count 0."""
    harness = GraphTestHarness()

    async def planner(state: AgentState) -> dict[str, object]:
        harness.planner_count += 1
        return {
            "plan": None,  # Simplified
            "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            "trace": (TraceEvent(node="planner", seconds=0.1),),
        }

    async def retrieve(state: AgentState) -> dict[str, object]:
        harness.retrieve_count += 1
        return {
            "sub_results": (),
            "usage": Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
            "trace": (TraceEvent(node="retrieve", seconds=0.2),),
        }

    async def synthesize(state: AgentState) -> dict[str, object]:
        harness.synthesize_count += 1
        return {
            "draft": "answer",
            "draft_refusal": False,
            "usage": Usage(input_tokens=30, output_tokens=15, cost_usd=0.003),
            "trace": (TraceEvent(node="synthesize", seconds=0.3),),
        }

    async def critic(state: AgentState) -> dict[str, object]:
        harness.critic_count += 1
        critique = Critique(verdict=CriticVerdict.PASS)
        return {
            "critique": critique,
            "critiques": (critique,),
            "usage": Usage(input_tokens=40, output_tokens=20, cost_usd=0.004),
            "trace": (TraceEvent(node="critic", seconds=0.4),),
        }

    graph = build_graph(
        planner=planner,
        retrieve=retrieve,
        synthesize=synthesize,
        critic=critic,
        max_revisions=5,
    )

    result = await graph.ainvoke(initial_state("q"))

    assert harness.planner_count == 1
    assert harness.retrieve_count == 1
    assert harness.synthesize_count == 1
    assert harness.critic_count == 1
    assert result["revision_count"] == 0
    assert result["critique"].verdict == CriticVerdict.PASS


@pytest.mark.asyncio
async def test_revision_cap_adversarial() -> None:
    """HARD CAP: critic ALWAYS REVISE, max_revisions=2.

    Guarantees synthesize runs exactly 3 times (initial + 2 revisions),
    critic runs exactly 3 times, and graph terminates with
    revision_count=2, critique.verdict=REVISE.
    """
    harness = GraphTestHarness()

    async def planner(state: AgentState) -> dict[str, object]:
        harness.planner_count += 1
        return {
            "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            "trace": (TraceEvent(node="planner", seconds=0.1),),
        }

    async def retrieve(state: AgentState) -> dict[str, object]:
        harness.retrieve_count += 1
        return {
            "sub_results": (),
            "usage": Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
            "trace": (TraceEvent(node="retrieve", seconds=0.2),),
        }

    async def synthesize(state: AgentState) -> dict[str, object]:
        harness.synthesize_count += 1
        # Increment revision_count when re-entering (critique is not None)
        revision_increment = 1 if state["critique"] is not None else 0
        return {
            "draft": f"answer_v{state['revision_count'] + revision_increment}",
            "draft_refusal": False,
            "revision_count": state["revision_count"] + revision_increment,
            "usage": Usage(input_tokens=30, output_tokens=15, cost_usd=0.003),
            "trace": (TraceEvent(node="synthesize", seconds=0.3),),
        }

    async def critic(state: AgentState) -> dict[str, object]:
        harness.critic_count += 1
        critique = Critique(verdict=CriticVerdict.REVISE)
        return {
            "critique": critique,
            "critiques": (critique,),
            "usage": Usage(input_tokens=40, output_tokens=20, cost_usd=0.004),
            "trace": (TraceEvent(node="critic", seconds=0.4),),
        }

    graph = build_graph(
        planner=planner,
        retrieve=retrieve,
        synthesize=synthesize,
        critic=critic,
        max_revisions=2,
    )

    result = await graph.ainvoke(initial_state("q"))

    # Synthesize: initial run + 2 re-entries = 3 times
    assert harness.synthesize_count == 3, f"Expected 3, got {harness.synthesize_count}"
    # Critic: one per synthesize run = 3 times
    assert harness.critic_count == 3, f"Expected 3, got {harness.critic_count}"
    # Final state
    assert result["revision_count"] == 2
    assert result["critique"].verdict == CriticVerdict.REVISE


@pytest.mark.asyncio
async def test_revise_once_then_pass() -> None:
    """Synthesize twice (initial + 1 revision), revision_count 1, final PASS."""
    harness = GraphTestHarness()

    async def planner(state: AgentState) -> dict[str, object]:
        harness.planner_count += 1
        return {
            "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            "trace": (TraceEvent(node="planner", seconds=0.1),),
        }

    async def retrieve(state: AgentState) -> dict[str, object]:
        harness.retrieve_count += 1
        return {
            "sub_results": (),
            "usage": Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
            "trace": (TraceEvent(node="retrieve", seconds=0.2),),
        }

    async def synthesize(state: AgentState) -> dict[str, object]:
        harness.synthesize_count += 1
        revision_increment = 1 if state["critique"] is not None else 0
        return {
            "draft": f"answer_v{state['revision_count'] + revision_increment}",
            "draft_refusal": False,
            "revision_count": state["revision_count"] + revision_increment,
            "usage": Usage(input_tokens=30, output_tokens=15, cost_usd=0.003),
            "trace": (TraceEvent(node="synthesize", seconds=0.3),),
        }

    async def critic(state: AgentState) -> dict[str, object]:
        harness.critic_count += 1
        # First critique: REVISE; second: PASS
        verdict = CriticVerdict.REVISE if harness.critic_count == 1 else CriticVerdict.PASS
        critique = Critique(verdict=verdict)
        return {
            "critique": critique,
            "critiques": (critique,),
            "usage": Usage(input_tokens=40, output_tokens=20, cost_usd=0.004),
            "trace": (TraceEvent(node="critic", seconds=0.4),),
        }

    graph = build_graph(
        planner=planner,
        retrieve=retrieve,
        synthesize=synthesize,
        critic=critic,
        max_revisions=5,
    )

    result = await graph.ainvoke(initial_state("q"))

    assert harness.synthesize_count == 2
    assert harness.critic_count == 2
    assert result["revision_count"] == 1
    assert result["critique"].verdict == CriticVerdict.PASS
    # Critiques channel should hold both
    assert len(result["critiques"]) == 2
    assert result["critiques"][0].verdict == CriticVerdict.REVISE
    assert result["critiques"][1].verdict == CriticVerdict.PASS


@pytest.mark.asyncio
async def test_accumulation_usage_and_trace() -> None:
    """Usage accumulates via reducer; trace preserves execution order."""
    harness = GraphTestHarness()

    async def planner(state: AgentState) -> dict[str, object]:
        harness.planner_count += 1
        return {
            "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            "trace": (TraceEvent(node="planner", seconds=0.1),),
        }

    async def retrieve(state: AgentState) -> dict[str, object]:
        harness.retrieve_count += 1
        return {
            "sub_results": (),
            "usage": Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
            "trace": (TraceEvent(node="retrieve", seconds=0.2),),
        }

    async def synthesize(state: AgentState) -> dict[str, object]:
        harness.synthesize_count += 1
        return {
            "draft": "answer",
            "draft_refusal": False,
            "usage": Usage(input_tokens=30, output_tokens=15, cost_usd=0.003),
            "trace": (TraceEvent(node="synthesize", seconds=0.3),),
        }

    async def critic(state: AgentState) -> dict[str, object]:
        harness.critic_count += 1
        critique = Critique(verdict=CriticVerdict.PASS)
        return {
            "critique": critique,
            "critiques": (critique,),
            "usage": Usage(input_tokens=40, output_tokens=20, cost_usd=0.004),
            "trace": (TraceEvent(node="critic", seconds=0.4),),
        }

    graph = build_graph(
        planner=planner,
        retrieve=retrieve,
        synthesize=synthesize,
        critic=critic,
        max_revisions=5,
    )

    result = await graph.ainvoke(initial_state("q"))

    # Usage should be summed via operator.add reducer
    total_usage = result["usage"]
    expected_input = 10 + 20 + 30 + 40
    expected_output = 5 + 10 + 15 + 20
    expected_cost = 0.001 + 0.002 + 0.003 + 0.004
    assert total_usage.input_tokens == expected_input
    assert total_usage.output_tokens == expected_output
    assert total_usage.cost_usd == pytest.approx(expected_cost, rel=1e-5)

    # Trace should preserve node execution order
    trace = result["trace"]
    assert len(trace) == 4
    assert trace[0].node == "planner"
    assert trace[1].node == "retrieve"
    assert trace[2].node == "synthesize"
    assert trace[3].node == "critic"


@pytest.mark.asyncio
async def test_max_revisions_zero() -> None:
    """max_revisions=0: critic REVISE still finalizes immediately.

    Synthesize runs once (no re-entries possible), revision_count stays 0,
    but critique verdict is REVISE and graph terminates.
    """
    harness = GraphTestHarness()

    async def planner(state: AgentState) -> dict[str, object]:
        harness.planner_count += 1
        return {
            "usage": Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            "trace": (TraceEvent(node="planner", seconds=0.1),),
        }

    async def retrieve(state: AgentState) -> dict[str, object]:
        harness.retrieve_count += 1
        return {
            "sub_results": (),
            "usage": Usage(input_tokens=20, output_tokens=10, cost_usd=0.002),
            "trace": (TraceEvent(node="retrieve", seconds=0.2),),
        }

    async def synthesize(state: AgentState) -> dict[str, object]:
        harness.synthesize_count += 1
        return {
            "draft": "answer",
            "draft_refusal": False,
            "usage": Usage(input_tokens=30, output_tokens=15, cost_usd=0.003),
            "trace": (TraceEvent(node="synthesize", seconds=0.3),),
        }

    async def critic(state: AgentState) -> dict[str, object]:
        harness.critic_count += 1
        critique = Critique(verdict=CriticVerdict.REVISE)
        return {
            "critique": critique,
            "critiques": (critique,),
            "usage": Usage(input_tokens=40, output_tokens=20, cost_usd=0.004),
            "trace": (TraceEvent(node="critic", seconds=0.4),),
        }

    graph = build_graph(
        planner=planner,
        retrieve=retrieve,
        synthesize=synthesize,
        critic=critic,
        max_revisions=0,
    )

    result = await graph.ainvoke(initial_state("q"))

    # Synthesize runs once (no revisions possible)
    assert harness.synthesize_count == 1
    # Critic runs once
    assert harness.critic_count == 1
    # revision_count stays 0
    assert result["revision_count"] == 0
    # But critique is REVISE (caveat case)
    assert result["critique"].verdict == CriticVerdict.REVISE
