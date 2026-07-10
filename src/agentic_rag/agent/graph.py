"""LangGraph-orchestrated agent loop: planner -> retrieve -> synthesize -> critic.

``build_graph`` composes the DAG and routing over caller-supplied node
callbacks; ``AgenticPipeline`` is the facade that binds the real nodes
(plan_query, gather, synthesize_draft, critique_draft) to a provider,
retriever, and reranker, and maps the final graph state to an ``AgentAnswer``.

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

import time
from collections.abc import Awaitable, Callable
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agent.critic import critique_draft
from agentic_rag.agent.gather import gather
from agentic_rag.agent.planner import plan_query
from agentic_rag.agent.state import (
    AgentAnswer,
    AgentState,
    CriticVerdict,
    Critique,
    TraceEvent,
    initial_state,
)
from agentic_rag.agent.synthesizer import synthesize_draft
from agentic_rag.config import Settings
from agentic_rag.observability import set_usage_attributes, tracer
from agentic_rag.pipeline.base import Answer, StageTiming
from agentic_rag.pipeline.citations import resolve_citations
from agentic_rag.pipeline.pipeline import SupportsRetrieve
from agentic_rag.providers.base import LLMProvider
from agentic_rag.rerank.base import Reranker
from agentic_rag.retrieval.base import RetrievalMode

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


class AgenticPipeline:
    """Agentic counterpart to ``RAGPipeline``: same constructor shape, and the
    returned ``AgentAnswer.answer`` is a vanilla ``Answer`` so CLI and eval
    consumers reuse their existing rendering paths.

    A refusal draft skips the critic (refusals are handled by refusal
    correctness counts per ADR-006, not by revision). ``caveat`` is set when
    the loop finalized while the last critique still said revise.
    """

    def __init__(
        self, retriever: SupportsRetrieve, reranker: Reranker, llm: LLMProvider, settings: Settings
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.settings = settings

    async def ask(
        self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> AgentAnswer:
        """Run the full agent loop on one question.

        The graph is compiled per call so the node closures can capture
        ``mode``; compilation is cheap relative to a single LLM call.
        """
        settings = self.settings

        async def planner_node(state: AgentState) -> dict[str, object]:
            start = time.perf_counter()
            with tracer().start_as_current_span("agent.plan") as plan_span:
                result = await plan_query(
                    self.llm,
                    state["question"],
                    max_sub_queries=settings.agent.max_sub_queries,
                    max_tokens=settings.agent.planner_max_tokens,
                    prompt_version=settings.agent.planner_prompt_version,
                )
                plan_span.set_attribute("agent.plan.kind", result.plan.kind.value)
                plan_span.set_attribute("agent.plan.sub_queries", len(result.plan.sub_queries))
                plan_span.set_attribute("agent.plan.fallback", result.fallback)
                set_usage_attributes(plan_span, result.usage)

            elapsed = time.perf_counter() - start
            event = TraceEvent(
                node="planner",
                seconds=elapsed,
                payload={
                    "question": state["question"],
                    "kind": result.plan.kind.value,
                    "sub_queries": list(result.plan.sub_queries),
                    "fallback": result.fallback,
                    "prompt_id": result.prompt_id,
                    "raw": result.raw,
                },
            )
            return {"plan": result.plan, "usage": result.usage, "trace": (event,)}

        async def retrieve_node(state: AgentState) -> dict[str, object]:
            plan = state["plan"]
            if plan is None:  # pragma: no cover - planner always precedes retrieve
                raise RuntimeError("retrieve node ran before planner")
            start = time.perf_counter()
            with tracer().start_as_current_span("agent.gather") as gather_span:
                result = await gather(
                    self.retriever,
                    self.reranker,
                    plan,
                    mode=mode,
                    candidate_pool=settings.rerank.candidate_pool,
                    top_k=settings.rerank.top_k,
                    max_context_tokens=settings.synthesis.max_context_tokens,
                    count_tokens=self.llm.count_tokens,
                )
                gather_span.set_attribute("agent.sub_queries", len(plan.sub_queries))
                gather_span.set_attribute("rag.context.tokens", result.context.token_count)
                gather_span.set_attribute("rag.chunks.count", len(result.context.chunks))
                set_usage_attributes(gather_span, result.usage)

            elapsed = time.perf_counter() - start
            event = TraceEvent(
                node="retrieve",
                seconds=elapsed,
                payload={
                    "reranked_chunk_ids": {
                        r.query: [c.chunk.chunk_id for c in r.chunks] for r in result.sub_results
                    },
                    "context_chunk_ids": [c.chunk.chunk_id for c in result.context.chunks],
                    "context_tokens": result.context.token_count,
                },
            )
            return {
                "sub_results": result.sub_results,
                "context": result.context,
                "usage": result.usage,
                "trace": (event,),
            }

        async def synthesize_node(state: AgentState) -> dict[str, object]:
            context = state["context"]
            if context is None:  # pragma: no cover - retrieve always precedes synthesize
                raise RuntimeError("synthesize node ran before retrieve")
            critique = state["critique"]
            revision = critique is not None
            revision_count = state["revision_count"] + (1 if revision else 0)
            start = time.perf_counter()
            with tracer().start_as_current_span("agent.synthesize") as synthesize_span:
                draft = await synthesize_draft(
                    self.llm,
                    state["question"],
                    context,
                    prior_draft=state["draft"] if revision else None,
                    critique=critique if revision else None,
                    max_tokens=settings.synthesis.max_answer_tokens,
                    prompt_version=settings.agent.synthesis_prompt_version,
                )
                synthesize_span.set_attribute("agent.revision", revision_count)
                synthesize_span.set_attribute("rag.refusal", draft.refusal)
                synthesize_span.set_attribute("agent.stray_sentinel", draft.stray_sentinel)
                set_usage_attributes(synthesize_span, draft.usage)

            elapsed = time.perf_counter() - start
            event = TraceEvent(
                node="synthesize",
                seconds=elapsed,
                payload={
                    "revision": revision_count,
                    "refusal": draft.refusal,
                    "stray_sentinel": draft.stray_sentinel,
                    "prompt_id": draft.prompt_id,
                    "draft": draft.text,
                },
            )
            return {
                "draft": draft.text,
                "draft_refusal": draft.refusal,
                "revision_count": revision_count,
                "usage": draft.usage,
                "trace": (event,),
            }

        async def critic_node(state: AgentState) -> dict[str, object]:
            if state["draft_refusal"]:
                with tracer().start_as_current_span("agent.critic") as critic_span:
                    critic_span.set_attribute("agent.skipped", True)

                critique = Critique(verdict=CriticVerdict.PASS)
                event = TraceEvent(node="critic", seconds=0.0, payload={"skipped": "refusal draft"})
                return {"critique": critique, "critiques": (critique,), "trace": (event,)}
            context = state["context"]
            if context is None:  # pragma: no cover - retrieve always precedes critic
                raise RuntimeError("critic node ran before retrieve")
            start = time.perf_counter()
            with tracer().start_as_current_span("agent.critic") as critic_span:
                result = await critique_draft(
                    self.llm,
                    state["question"],
                    context,
                    state["draft"],
                    max_tokens=settings.agent.critic_max_tokens,
                    prompt_version=settings.agent.critic_prompt_version,
                )
                critic_span.set_attribute("agent.verdict", result.critique.verdict.value)
                critic_span.set_attribute("agent.issues", len(result.critique.issues))
                critic_span.set_attribute("agent.skipped", False)
                set_usage_attributes(critic_span, result.usage)

            elapsed = time.perf_counter() - start
            event = TraceEvent(
                node="critic",
                seconds=elapsed,
                payload={
                    "verdict": result.critique.verdict.value,
                    "issues": [
                        {"kind": i.kind.value, "detail": i.detail, "fix": i.fix}
                        for i in result.critique.issues
                    ],
                    "fallback": result.fallback,
                    "prompt_id": result.prompt_id,
                },
            )
            return {
                "critique": result.critique,
                "critiques": (result.critique,),
                "usage": result.usage,
                "trace": (event,),
            }

        graph = build_graph(
            planner=planner_node,
            retrieve=retrieve_node,
            synthesize=synthesize_node,
            critic=critic_node,
            max_revisions=settings.agent.max_revisions,
        )
        # ainvoke's declared return is dict[str, Any]; the channels are AgentState's
        final = cast(AgentState, await graph.ainvoke(initial_state(question)))

        plan = final["plan"]
        context = final["context"]
        if plan is None or context is None:  # pragma: no cover - graph always runs all nodes
            raise RuntimeError("graph finished without plan/context")

        citation_result = resolve_citations(final["draft"], context.chunks)
        critiques = final["critiques"]
        caveat = bool(critiques) and critiques[-1].verdict is CriticVerdict.REVISE

        # One StageTiming per node, summed across loop iterations
        node_seconds: dict[str, float] = {}
        for event in final["trace"]:
            node_seconds[event.node] = node_seconds.get(event.node, 0.0) + event.seconds
        timings = [StageTiming(stage, seconds) for stage, seconds in node_seconds.items()]

        answer = Answer(
            text=citation_result.text,
            citations=citation_result.citations,
            context=context.chunks,
            usage=final["usage"],
            timings=timings,
            refusal=final["draft_refusal"],
            invalid_citations=citation_result.invalid_markers,
        )
        return AgentAnswer(
            answer=answer,
            plan=plan,
            revisions=final["revision_count"],
            critiques=critiques,
            caveat=caveat,
            trace=final["trace"],
        )
