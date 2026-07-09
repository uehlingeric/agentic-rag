"""End-to-end AgenticPipeline tests: real nodes, scripted LLM, zero live calls.

Every LLM reply is played back FIFO through ``PlaybackLLM``, so these tests
exercise the full compiled graph — planner parse, per-sub-query gather,
synthesis sentinel handling, critic verdicts, revision routing, and the final
AgentAnswer assembly — deterministically.
"""

from __future__ import annotations

from dataclasses import replace

from agentic_rag.agent.graph import AgenticPipeline
from agentic_rag.agent.replay import PlaybackLLM
from agentic_rag.agent.state import CriticVerdict, PlanKind
from agentic_rag.config import Settings
from agentic_rag.providers.base import Completion, Usage
from agentic_rag.rerank.base import NoopReranker
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="sp800-53r5",
        section_id="AC-2",
        section_ids=["AC-2"],
        section_path="AC-2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        page_start=1,
        page_end=1,
        token_count=8,
        text=text,
    )


def scored(chunk: ChunkRecord, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank)


class StubRetriever:
    """Returns canned chunks per query string."""

    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.results[query]


def completion(text: str) -> Completion:
    return Completion(
        text=text,
        model="playback",
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        stop_reason="end_turn",
    )


def make_settings() -> Settings:
    return Settings()


async def test_multi_hop_pass_first_critique() -> None:
    """Multi-hop plan: 2 sub-queries gathered, draft passes critic, no revision."""
    plan_reply = (
        '{"classification": "multi_hop", '
        '"sub_queries": ["FIPS 199 objectives", "AC-2 requirements"]}'
    )
    draft_reply = "FIPS 199 defines objectives [1]. AC-2 manages accounts [2]."
    critic_reply = '{"verdict": "pass"}'
    llm = PlaybackLLM.from_completions(
        [completion(plan_reply), completion(draft_reply), completion(critic_reply)]
    )

    chunk_a = make_chunk("c-fips", "FIPS 199 defines objectives.")
    chunk_b = make_chunk("c-ac2", "AC-2 manages accounts.")
    retriever = StubRetriever(
        {
            "FIPS 199 objectives": [scored(chunk_a, 1)],
            "AC-2 requirements": [scored(chunk_b, 1)],
        }
    )

    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, make_settings())
    result = await pipeline.ask("How do FIPS 199 objectives relate to AC-2?")

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert retriever.queries == ["FIPS 199 objectives", "AC-2 requirements"]
    assert result.revisions == 0
    assert result.caveat is False
    assert result.answer.refusal is False
    # Citations resolve against the merged context: [1]=c-fips, [2]=c-ac2
    assert [c.chunk.chunk_id for c in result.answer.citations] == ["c-fips", "c-ac2"]
    assert result.answer.invalid_citations == []
    # 3 LLM calls (planner, synthesize, critic), all usage accumulated
    assert llm.remaining == 0
    assert result.answer.usage.input_tokens == 300
    # Trace covers each node once, in execution order
    assert [e.node for e in result.trace] == ["planner", "retrieve", "synthesize", "critic"]


async def test_revision_loop_then_pass() -> None:
    """Critic revise -> synthesizer revises once -> critic passes."""
    plan_reply = '{"classification": "direct"}'
    first_draft = "Accounts are managed."  # uncited claim
    critic_revise = (
        '{"verdict": "revise", "issues": [{"kind": "uncited_claim", '
        '"detail": "The sentence has no citation.", "fix": "Cite excerpt [1]."}]}'
    )
    revised_draft = "Accounts are managed [1]."
    critic_pass = '{"verdict": "pass"}'
    llm = PlaybackLLM.from_completions(
        [
            completion(plan_reply),
            completion(first_draft),
            completion(critic_revise),
            completion(revised_draft),
            completion(critic_pass),
        ]
    )

    question = "What does AC-2 require?"
    chunk = make_chunk("c-ac2", "AC-2 manages accounts.")
    retriever = StubRetriever({question: [scored(chunk, 1)]})

    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, make_settings())
    result = await pipeline.ask(question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.revisions == 1
    assert result.caveat is False
    assert result.answer.text == "Accounts are managed [1]."
    assert [c.verdict for c in result.critiques] == [CriticVerdict.REVISE, CriticVerdict.PASS]
    assert llm.remaining == 0


async def test_revision_cap_finalizes_with_caveat() -> None:
    """Critic never passes: exactly max_revisions rewrites, then caveat=True."""
    plan_reply = '{"classification": "direct"}'
    critic_revise = (
        '{"verdict": "revise", "issues": [{"kind": "incomplete", '
        '"detail": "Half the question is unanswered.", "fix": "Answer both parts."}]}'
    )
    # planner + (draft, critique) x 3: initial + 2 revisions, critic revises each time
    llm = PlaybackLLM.from_completions(
        [
            completion(plan_reply),
            completion("Draft one [1]."),
            completion(critic_revise),
            completion("Draft two [1]."),
            completion(critic_revise),
            completion("Draft three [1]."),
            completion(critic_revise),
        ]
    )

    question = "What does AC-2 require and how often is it reviewed?"
    chunk = make_chunk("c-ac2", "AC-2 manages accounts.")
    retriever = StubRetriever({question: [scored(chunk, 1)]})

    settings = make_settings()
    assert settings.agent.max_revisions == 2
    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, settings)
    result = await pipeline.ask(question)

    assert result.revisions == 2
    assert result.caveat is True
    assert result.answer.text == "Draft three [1]."
    assert len(result.critiques) == 3
    assert all(c.verdict == CriticVerdict.REVISE for c in result.critiques)
    assert llm.remaining == 0


async def test_refusal_skips_critic() -> None:
    """A refusal draft finalizes without a critic LLM call."""
    plan_reply = '{"classification": "direct"}'
    refusal_reply = "[NO_ANSWER] The excerpts do not state the password length."
    llm = PlaybackLLM.from_completions([completion(plan_reply), completion(refusal_reply)])

    question = "What password length does AC-2 require?"
    chunk = make_chunk("c-ac2", "AC-2 manages accounts.")
    retriever = StubRetriever({question: [scored(chunk, 1)]})

    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, make_settings())
    result = await pipeline.ask(question)

    assert result.answer.refusal is True
    assert result.revisions == 0
    assert result.caveat is False
    # Only planner + synthesize consumed a playback slot; critic was skipped
    assert llm.remaining == 0
    critic_events = [e for e in result.trace if e.node == "critic"]
    assert critic_events[0].payload == {"skipped": "refusal draft"}


async def test_invalid_citation_marker_stripped() -> None:
    """A marker beyond the merged context is stripped and reported."""
    plan_reply = '{"classification": "direct"}'
    draft_reply = "Accounts are managed [1]. Reviews happen yearly [7]."
    critic_pass = '{"verdict": "pass"}'
    llm = PlaybackLLM.from_completions(
        [completion(plan_reply), completion(draft_reply), completion(critic_pass)]
    )

    question = "What does AC-2 require?"
    chunk = make_chunk("c-ac2", "AC-2 manages accounts.")
    retriever = StubRetriever({question: [scored(chunk, 1)]})

    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, make_settings())
    result = await pipeline.ask(question)

    assert result.answer.invalid_citations == [7]
    assert "[7]" not in result.answer.text
    assert [c.chunk.chunk_id for c in result.answer.citations] == ["c-ac2"]


async def test_dedupe_across_sub_queries_end_to_end() -> None:
    """A chunk surfaced by both sub-queries appears once in the merged context."""
    plan_reply = '{"classification": "multi_hop", "sub_queries": ["query one", "query two"]}'
    draft_reply = "Shared fact [1]. Second fact [2]."
    critic_pass = '{"verdict": "pass"}'
    llm = PlaybackLLM.from_completions(
        [completion(plan_reply), completion(draft_reply), completion(critic_pass)]
    )

    shared = make_chunk("c-shared", "A shared chunk.")
    other = make_chunk("c-other", "Another chunk.")
    retriever = StubRetriever(
        {
            "query one": [scored(shared, 1)],
            "query two": [scored(replace(shared, token_count=8), 1), scored(other, 2)],
        }
    )

    pipeline = AgenticPipeline(retriever, NoopReranker(), llm, make_settings())
    result = await pipeline.ask("How do the two facts relate?")

    assert [c.chunk.chunk_id for c in result.answer.context] == ["c-shared", "c-other"]
