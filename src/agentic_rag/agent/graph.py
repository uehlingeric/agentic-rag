"""LangGraph-orchestrated agent loop: planner -> retrieve -> synthesize -> critic.

LangGraph wiring for the agentic answer loop. Nodes are callback functions
supplied by the caller (e.g., an AgenticPipeline); this module only composes
the DAG and routing logic.

Orchestration only: all LLM calls stay behind the ``LLMProvider`` protocol
(ADR-001, ADR-007). Contracts in ``agentic_rag.agent.state``.

Revision routing: the critic node is responsible for returning a ``critique``
(last-write-wins channel). When ``critique`` is not None AND its ``verdict``
is ``CriticVerdict.REVISE`` AND ``revision_count < max_revisions``, the graph
routes back to synthesize (treating this as a re-entry for one more iteration).
Otherwise it routes to END. Per the convention: the synthesize node increments
``revision_count`` when it runs with a non-None critique input, so the graph
guarantees at most ``max_revisions`` re-entries into synthesize as long as
synthesize follows this convention. The graph does not enforce the convention;
it only routes on the channel state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agent.state import AgentState, CriticVerdict

NodeFn = Callable[[AgentState], Awaitable[dict[str, object]]]


def build_graph(
    *,
    planner: NodeFn,
    retrieve: NodeFn,
    synthesize: NodeFn,
    critic: NodeFn,
    max_revisions: int,
) -> CompiledStateGraph[AgentState]:
    """Compile the agent loop graph.

    Args:
        planner: Node function: AgentState -> dict[str, object].
        retrieve: Node function: AgentState -> dict[str, object].
        synthesize: Node function: AgentState -> dict[str, object].
        critic: Node function: AgentState -> dict[str, object].
        max_revisions: Maximum times synthesize re-enters after critic REVISE.

    Returns:
        Compiled graph ready for ainvoke(initial_state(...)).
    """
    graph: StateGraph[AgentState] = StateGraph(AgentState)

    graph.add_node("planner", planner)  # type: ignore[call-overload]
    graph.add_node("retrieve", retrieve)  # type: ignore[call-overload]
    graph.add_node("synthesize", synthesize)  # type: ignore[call-overload]
    graph.add_node("critic", critic)  # type: ignore[call-overload]

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critic")

    def route_critic(state: AgentState) -> str:
        """Route from critic: revise or finalize.

        Returns "revise" if critique is not None AND verdict is REVISE AND
        revision_count < max_revisions; else "finalize".
        """
        critique = state.get("critique")
        if (
            critique is not None
            and critique.verdict == CriticVerdict.REVISE
            and state["revision_count"] < max_revisions
        ):
            return "revise"
        return "finalize"

    graph.add_conditional_edges(
        "critic",
        route_critic,
        {
            "revise": "synthesize",
            "finalize": END,
        },
    )

    return graph.compile()
