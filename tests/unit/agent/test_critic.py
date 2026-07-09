"""Tests for LLM-as-critic draft revision guidance."""

from __future__ import annotations

import pytest

from agentic_rag.agent.critic import critique_draft
from agentic_rag.agent.state import CriticVerdict, IssueKind
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.providers.base import Completion, Message, Role, Usage
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


class StubLLM:
    """Stub LLM provider for testing: pops replies from a list and records calls."""

    def __init__(self, replies: list[str], name: str = "stub-provider") -> None:
        """Initialize with a list of reply texts to return in order.

        Args:
            replies: List of text responses to return on successive complete() calls.
            name: Name of the provider.
        """
        self._replies = list(replies)
        self._name = name
        self.recorded_messages: list[list[Message]] = []
        self.recorded_models: list[str | None] = []
        self.recorded_max_tokens: list[int] = []
        self.recorded_temps: list[float] = []

    @property
    def name(self) -> str:
        return self._name

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Pop the next reply, record call details, and return completion."""
        if not self._replies:
            raise RuntimeError("No more replies available for StubLLM")
        text = self._replies.pop(0)
        self.recorded_messages.append(list(messages))
        self.recorded_models.append(model)
        self.recorded_max_tokens.append(max_tokens)
        self.recorded_temps.append(temperature)
        return Completion(
            text=text,
            model="stub-model",
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        )

    def stream(self, *args, **kwargs):
        """Not implemented for critic tests."""
        raise NotImplementedError("stream() not implemented in StubLLM")

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split())


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    heading: str = "AC-2 ACCOUNT MANAGEMENT",
    text: str = "The organization manages system accounts.",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=heading,
        heading=heading,
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_context(chunks: list[ChunkRecord]) -> BuiltContext:
    """Helper to create a BuiltContext from chunks."""
    lines = []
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        marker = i + 1
        excerpt = (
            f"[{marker}] {chunk.doc_id} §{chunk.section_id} — "
            f"{chunk.heading} (p.{chunk.page_start})\n{chunk.text}\n"
        )
        lines.append(excerpt)
        scored_chunks.append(ScoredChunk(chunk=chunk, score=1.0 - i * 0.1, rank=marker))
    text = "".join(lines)
    return BuiltContext(text=text, chunks=scored_chunks, token_count=len(text.split()))


# Offline tests (all pass without external providers)


async def test_critic_pass() -> None:
    """Happy path: {\"verdict\": \"pass\"} → PASS, no issues, fallback False."""
    reply = '{"verdict": "pass"}'
    llm = StubLLM([reply], name="test-critic")
    chunk = make_chunk("c1", text="Account management text.")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="What is account management?",
        context=context,
        draft="The organization manages accounts [1].",
    )

    assert result.critique.verdict == CriticVerdict.PASS
    assert result.critique.issues == ()
    assert result.fallback is False
    assert len(llm.recorded_messages) == 1
    assert result.prompt_id == "critic.v1"
    assert result.raw == reply


async def test_critic_revise_two_issues() -> None:
    """Revise with two issues (different kinds) → typed CriticIssue tuple."""
    reply = (
        '{"verdict": "revise", "issues": ['
        '{"kind": "uncited_claim", "detail": "The first part lacks citation.", '
        '"fix": "Add [1] after the claim."},'
        '{"kind": "unsupported_citation", '
        '"detail": "Excerpt 2 does not discuss password requirements.", '
        '"fix": "Remove [2] or use [1] instead."}'
        "]}"
    )
    llm = StubLLM([reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="How is account management handled?",
        context=context,
        draft="First claim. Second claim [2].",
    )

    assert result.critique.verdict == CriticVerdict.REVISE
    assert len(result.critique.issues) == 2
    assert result.critique.issues[0].kind == IssueKind.UNCITED_CLAIM
    assert result.critique.issues[0].detail == "The first part lacks citation."
    assert result.critique.issues[0].fix == "Add [1] after the claim."
    assert result.critique.issues[1].kind == IssueKind.UNSUPPORTED_CITATION
    assert result.critique.issues[1].detail == "Excerpt 2 does not discuss password requirements."
    assert result.critique.issues[1].fix == "Remove [2] or use [1] instead."
    assert result.fallback is False


async def test_critic_planted_bad_citation() -> None:
    """Planted-bad-citation: draft cites wrong excerpt for a claim."""
    reply = (
        '{"verdict": "revise", "issues": [{"kind": "unsupported_citation", '
        '"detail": "The claim \'password rotation\' is not in excerpt [1].", '
        '"fix": "Remove [1] or use a different excerpt."}]}'
    )
    llm = StubLLM([reply])
    chunk = make_chunk(
        "ac2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        text="Organizations manage system accounts including creation and removal.",
    )
    context = make_context([chunk])
    draft = "Password rotation is required every 90 days [1]."

    result = await critique_draft(
        llm,
        question="What are password requirements?",
        context=context,
        draft=draft,
    )

    assert result.critique.verdict == CriticVerdict.REVISE
    assert len(result.critique.issues) >= 1
    assert result.critique.issues[0].kind == IssueKind.UNSUPPORTED_CITATION
    assert "password" in result.critique.issues[0].detail.lower()


async def test_critic_json_in_code_fences() -> None:
    """JSON wrapped in code fences and prose is still parsed."""
    reply = """Some analysis.
```json
{"verdict": "pass"}
```
No issues found."""
    llm = StubLLM([reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Is this correct?",
        context=context,
        draft="A claim [1].",
    )

    assert result.critique.verdict == CriticVerdict.PASS


async def test_critic_repair_on_first_malformed() -> None:
    """First reply malformed, second valid; usage sums both calls."""
    bad_reply = "This is not JSON at all."
    good_reply = (
        '{"verdict": "revise", "issues": '
        '[{"kind": "uncited_claim", "detail": "Missing citation.", "fix": "Add [1]."}]}'
    )
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Is this good?",
        context=context,
        draft="A claim without citation.",
        max_parse_retries=2,
    )

    assert result.critique.verdict == CriticVerdict.REVISE
    assert result.fallback is False
    # Two calls: combined usage
    assert result.usage.input_tokens == 200  # 100 + 100
    assert result.usage.output_tokens == 100  # 50 + 50
    assert result.usage.cost_usd == 0.002  # 0.001 + 0.001
    # Second call's messages include the bad reply and repair instruction
    assert len(llm.recorded_messages) == 2
    second_call_msgs = llm.recorded_messages[1]
    assert len(second_call_msgs) >= 3
    assert second_call_msgs[-2].role == Role.ASSISTANT
    assert second_call_msgs[-2].content == bad_reply
    assert second_call_msgs[-1].role == Role.USER
    assert "JSON object" in second_call_msgs[-1].content


async def test_critic_all_malformed_fallback() -> None:
    """All replies malformed → PASS fallback True, usage summed."""
    bad1 = "not json"
    bad2 = "{incomplete"
    llm = StubLLM([bad1, bad2])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Is this good?",
        context=context,
        draft="Some draft.",
        max_parse_retries=1,  # 2 total calls
    )

    # Should have made exactly 2 calls (1 initial + 1 retry)
    assert len(llm.recorded_messages) == 2
    # Fallback to PASS
    assert result.critique.verdict == CriticVerdict.PASS
    assert result.critique.issues == ()
    assert result.fallback is True
    # Usage summed
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 100


async def test_critic_revise_empty_issues_parse_failure() -> None:
    """verdict \"revise\" with empty issues list → treated as parse failure."""
    bad_reply = '{"verdict": "revise", "issues": []}'
    good_reply = '{"verdict": "pass"}'
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check this.",
        context=context,
        draft="A draft.",
        max_parse_retries=2,
    )

    # Should have triggered repair on first failure
    assert len(llm.recorded_messages) == 2
    assert result.critique.verdict == CriticVerdict.PASS
    assert result.fallback is False


async def test_critic_unknown_issue_kind() -> None:
    """Unknown issue kind → parse failure → repair."""
    bad_reply = (
        '{"verdict": "revise", "issues": '
        '[{"kind": "unknown_kind", "detail": "Bad.", "fix": "Fix it."}]}'
    )
    good_reply = '{"verdict": "pass"}'
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
        max_parse_retries=2,
    )

    assert len(llm.recorded_messages) == 2
    assert result.critique.verdict == CriticVerdict.PASS
    assert result.fallback is False


async def test_critic_pass_with_stray_issues() -> None:
    """\"pass\" with stray issues present → PASS with no issues, not a failure."""
    reply = (
        '{"verdict": "pass", "issues": '
        '[{"kind": "uncited_claim", "detail": "Ignored.", "fix": "Ignored."}]}'
    )
    llm = StubLLM([reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
    )

    assert result.critique.verdict == CriticVerdict.PASS
    assert result.critique.issues == ()
    assert result.fallback is False
    # Only one call (no repair needed)
    assert len(llm.recorded_messages) == 1


async def test_critic_prompt_integrity() -> None:
    """Prompt integrity: load_prompt renders without KeyError, contains draft."""
    reply = '{"verdict": "pass"}'
    llm = StubLLM([reply])
    chunk = make_chunk("c1", text="Some context text.")
    context = make_context([chunk])
    question = "What is the question?"
    draft = "This is the draft answer [1]."

    result = await critique_draft(
        llm,
        question=question,
        context=context,
        draft=draft,
    )

    # Verify the prompt loaded and rendered without KeyError
    assert result.prompt_id == "critic.v1"

    # Verify the rendered prompt is in the first message
    first_msg = llm.recorded_messages[0][0]
    assert draft in first_msg.content
    assert question in first_msg.content
    assert "Some context text." in first_msg.content


async def test_critic_whitespace_stripped() -> None:
    """Detail and fix fields have whitespace stripped."""
    reply = """{
    "verdict": "revise",
    "issues": [
        {"kind": "uncited_claim", "detail": "  Uncited claim.  ", "fix": "  Add citation.  "}
    ]
}"""
    llm = StubLLM([reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
    )

    assert result.critique.issues[0].detail == "Uncited claim."
    assert result.critique.issues[0].fix == "Add citation."


async def test_critic_issues_empty_detail_parse_failure() -> None:
    """Empty detail field → parse failure."""
    bad_reply = (
        '{"verdict": "revise", "issues": '
        '[{"kind": "uncited_claim", "detail": "", "fix": "Fix it."}]}'
    )
    good_reply = '{"verdict": "pass"}'
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
        max_parse_retries=1,
    )

    assert len(llm.recorded_messages) == 2
    assert result.fallback is False


async def test_critic_issues_empty_fix_parse_failure() -> None:
    """Empty fix field → parse failure."""
    bad_reply = (
        '{"verdict": "revise", "issues": [{"kind": "uncited_claim", "detail": "Bad.", "fix": ""}]}'
    )
    good_reply = '{"verdict": "pass"}'
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
        max_parse_retries=1,
    )

    assert len(llm.recorded_messages) == 2
    assert result.fallback is False


async def test_critic_version_pinning() -> None:
    """prompt_version=1 pins to critic.v1."""
    reply = '{"verdict": "pass"}'
    llm = StubLLM([reply])
    chunk = make_chunk("c1")
    context = make_context([chunk])

    result = await critique_draft(
        llm,
        question="Check.",
        context=context,
        draft="Draft.",
        prompt_version=1,
    )

    assert result.prompt_id == "critic.v1"


# Live tests (marked to run manually with real providers)


@pytest.mark.live
async def test_critic_live_planted_bad_citation() -> None:
    """Live: draft citing wrong excerpt → verdict REVISE with unsupported_citation."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    # The critic runs on the pipeline's own generation provider
    critic = get_llm_provider(settings.provider, settings)

    # Build context with two excerpts
    ac2 = make_chunk(
        "ac2-chunk",
        doc_id="sp800-53r5",
        section_id="AC-2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        text=(
            "Organizations manage system accounts. Account management includes the creation, "
            "activation, modification, disablement, and removal of system accounts."
        ),
    )

    at2 = make_chunk(
        "at2-chunk",
        doc_id="sp800-53r5",
        section_id="AT-2",
        heading="AT-2 LITERACY TRAINING AND AWARENESS",
        text=(
            "Literacy training and awareness ensures personnel receive security and privacy "
            "training before authorizing access to the system."
        ),
    )

    context = make_context([ac2, at2])

    # Draft makes a claim about account management but cites the training excerpt
    draft = (
        "Organizations manage system accounts by creating, modifying, and removing "
        "accounts as needed [2], with accounts scoped to authorized personnel."
    )

    result = await critique_draft(
        critic,
        question="How does the organization manage system accounts?",
        context=context,
        draft=draft,
    )

    # Should detect the bad citation
    assert result.critique.verdict == CriticVerdict.REVISE
    assert len(result.critique.issues) >= 1
    # At least one issue should be unsupported_citation or uncited_claim
    kinds = {issue.kind for issue in result.critique.issues}
    assert IssueKind.UNSUPPORTED_CITATION in kinds or IssueKind.UNCITED_CLAIM in kinds


@pytest.mark.live
async def test_critic_live_clean_answer_passes() -> None:
    """Live: every sentence correctly cited → verdict PASS."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    # The critic runs on the pipeline's own generation provider
    critic = get_llm_provider(settings.provider, settings)

    # Build context with a single excerpt
    ac2 = make_chunk(
        "ac2-chunk",
        doc_id="sp800-53r5",
        section_id="AC-2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        text=(
            "Organizations manage system accounts. Account management includes the creation, "
            "activation, modification, disablement, and removal of system accounts. Accounts "
            "are scoped to individuals and service accounts authorized to access the system."
        ),
    )

    context = make_context([ac2])

    # Draft is well-cited and directly supported by the excerpt
    draft = (
        "Organizations manage system accounts [1]. Account management includes the creation "
        "and removal of accounts [1]. Accounts are scoped to authorized individuals [1]."
    )

    result = await critique_draft(
        critic,
        question="How does the organization manage system accounts?",
        context=context,
        draft=draft,
    )

    # Should pass without complaint
    assert result.critique.verdict == CriticVerdict.PASS
    assert result.critique.issues == ()
    assert result.fallback is False
