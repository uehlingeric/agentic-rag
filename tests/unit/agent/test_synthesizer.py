"""Tests for draft synthesis with optional revision via critique feedback."""

from __future__ import annotations

import pytest

from agentic_rag.agent.state import CriticIssue, CriticVerdict, Critique, IssueKind
from agentic_rag.agent.synthesizer import synthesize_draft
from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.providers.base import Completion, Message, Role, Usage
from agentic_rag.retrieval.base import ChunkRecord


def make_chunk(
    chunk_id: str,
    *,
    text: str = "Sample content",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="test",
        section_id="1",
        section_ids=["1"],
        section_path="Test",
        heading="Test",
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


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
        """Not implemented for synthesis tests."""
        raise NotImplementedError("stream() not implemented in StubLLM")

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split())


@pytest.mark.asyncio
async def test_first_pass_single_user_message() -> None:
    """First pass sends single USER message with agent-synthesis prompt."""
    llm = StubLLM(["An answer from context."])
    context = BuiltContext(text="[1] chunk content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
        model="test-model",
        max_tokens=512,
    )

    # Verify single message recorded
    assert len(llm.recorded_messages) == 1
    assert len(llm.recorded_messages[0]) == 1
    assert llm.recorded_messages[0][0].role == Role.USER
    assert "What is X?" in llm.recorded_messages[0][0].content
    assert "[1] chunk content" in llm.recorded_messages[0][0].content

    # Verify temperature was 0.0
    assert llm.recorded_temps[0] == 0.0

    # Verify result
    assert result.text == "An answer from context."
    assert result.refusal is False
    assert result.stray_sentinel is False
    assert result.usage.input_tokens == 100
    assert result.model == "stub-model"


@pytest.mark.asyncio
async def test_leading_sentinel_marks_refusal() -> None:
    """Reply starting with [NO_ANSWER] is marked as refusal."""
    reply = f"{NO_ANSWER_SENTINEL} The context does not contain relevant information."
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] unrelated content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
    )

    assert result.refusal is True
    assert result.text == "The context does not contain relevant information."
    assert result.stray_sentinel is False


@pytest.mark.asyncio
async def test_only_leading_sentinel_is_empty_refusal() -> None:
    """Reply that is only [NO_ANSWER] (with whitespace) is empty refusal."""
    reply = f"  {NO_ANSWER_SENTINEL}  "
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
    )

    assert result.refusal is True
    assert result.text == ""
    assert result.stray_sentinel is False


@pytest.mark.asyncio
async def test_stray_sentinel_without_refusal() -> None:
    """Reply ending with [NO_ANSWER] is marked with stray_sentinel, not refusal."""
    reply = f"Organizations must implement control measures [1]. {NO_ANSWER_SENTINEL}"
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] chunk content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
    )

    assert result.refusal is False
    assert result.stray_sentinel is True
    assert result.text == "Organizations must implement control measures [1]."


@pytest.mark.asyncio
async def test_stray_sentinel_with_whitespace() -> None:
    """Stray sentinel followed by whitespace is correctly stripped."""
    reply = f"Some answer [2].  {NO_ANSWER_SENTINEL}  \n"
    llm = StubLLM([reply])
    context = BuiltContext(text="[2] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What?",
        context=context,
    )

    assert result.refusal is False
    assert result.stray_sentinel is True
    assert result.text == "Some answer [2]."


@pytest.mark.asyncio
async def test_revision_pass_three_messages() -> None:
    """Revision pass sends USER, ASSISTANT (prior draft), USER (revise) messages."""
    prior_draft = "Initial answer [1]."
    critique = Critique(
        verdict=CriticVerdict.REVISE,
        issues=(
            CriticIssue(
                kind=IssueKind.UNCITED_CLAIM,
                detail="The claim about X is unsupported",
                fix="Add citation [2] or remove the claim",
            ),
        ),
    )

    llm = StubLLM(["Revised answer [1][2]."])
    context = BuiltContext(text="[1] chunk1\n[2] chunk2", chunks=[], token_count=10)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
        prior_draft=prior_draft,
        critique=critique,
    )

    # Verify 3 messages recorded
    assert len(llm.recorded_messages) == 1
    assert len(llm.recorded_messages[0]) == 3

    # Check message roles and content
    msg0 = llm.recorded_messages[0][0]
    assert msg0.role == Role.USER
    assert "What is X?" in msg0.content

    msg1 = llm.recorded_messages[0][1]
    assert msg1.role == Role.ASSISTANT
    assert msg1.content == prior_draft

    msg2 = llm.recorded_messages[0][2]
    assert msg2.role == Role.USER
    assert "uncited_claim" in msg2.content
    assert "Add citation [2] or remove the claim" in msg2.content

    assert result.text == "Revised answer [1][2]."


@pytest.mark.asyncio
async def test_revision_issues_formatted_correctly() -> None:
    """Critique issues are formatted as '- {kind}: {detail} Fix: {fix}' lines."""
    prior_draft = "Initial answer."
    critique = Critique(
        verdict=CriticVerdict.REVISE,
        issues=(
            CriticIssue(
                kind=IssueKind.UNSUPPORTED_CITATION,
                detail="Citation [1] does not support the claim",
                fix="Remove [1] or add a better citation",
            ),
            CriticIssue(
                kind=IssueKind.INCOMPLETE,
                detail="Missing information about Y",
                fix="Add information from the context about Y",
            ),
        ),
    )

    llm = StubLLM(["Fixed answer."])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    await synthesize_draft(
        llm,
        question="Q?",
        context=context,
        prior_draft=prior_draft,
        critique=critique,
    )

    # Extract the revise message
    revise_msg = llm.recorded_messages[0][2]
    issues_text = revise_msg.content

    # Verify formatting
    assert (
        "- unsupported_citation: Citation [1] does not support the claim "
        "Fix: Remove [1] or add a better citation"
    ) in issues_text
    assert (
        "- incomplete: Missing information about Y Fix: Add information from the context about Y"
    ) in issues_text


@pytest.mark.asyncio
async def test_prior_draft_without_critique_raises() -> None:
    """ValueError raised when prior_draft is provided but critique is None."""
    llm = StubLLM(["Should not be called"])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    with pytest.raises(
        ValueError, match="prior_draft and critique must both be None or both be provided"
    ):
        await synthesize_draft(
            llm,
            question="Q?",
            context=context,
            prior_draft="Initial answer.",
            critique=None,
        )


@pytest.mark.asyncio
async def test_critique_without_prior_draft_raises() -> None:
    """ValueError raised when critique is provided but prior_draft is None."""
    critique = Critique(verdict=CriticVerdict.REVISE, issues=())
    llm = StubLLM(["Should not be called"])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    with pytest.raises(
        ValueError, match="prior_draft and critique must both be None or both be provided"
    ):
        await synthesize_draft(
            llm,
            question="Q?",
            context=context,
            prior_draft=None,
            critique=critique,
        )


@pytest.mark.asyncio
async def test_temperature_zero_enforced() -> None:
    """Temperature is always 0.0 for deterministic generation."""
    llm = StubLLM(["Answer."])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    await synthesize_draft(
        llm,
        question="Q?",
        context=context,
        model="test-model",
    )

    assert llm.recorded_temps[0] == 0.0


@pytest.mark.asyncio
async def test_prompt_id_is_agent_synthesis() -> None:
    """Prompt ID is agent-synthesis.v2."""
    llm = StubLLM(["Answer."])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="Q?",
        context=context,
    )

    assert result.prompt_id == "agent-synthesis.v1"


@pytest.mark.asyncio
async def test_no_stray_sentinel_on_normal_answer() -> None:
    """Normal answer without sentinel has stray_sentinel=False."""
    llm = StubLLM(["Normal answer without any sentinel."])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="Q?",
        context=context,
    )

    assert result.refusal is False
    assert result.stray_sentinel is False
    assert result.text == "Normal answer without any sentinel."


@pytest.mark.asyncio
async def test_leading_and_stray_sentinel_both_present() -> None:
    """If answer starts with sentinel, leading takes precedence (refusal=True)."""
    # Edge case: starts and ends with sentinel
    reply = f"{NO_ANSWER_SENTINEL} some text {NO_ANSWER_SENTINEL}"
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="Q?",
        context=context,
    )

    # Leading sentinel makes it a refusal
    assert result.refusal is True
    # After stripping leading, remaining text is "some text [NO_ANSWER]"
    # Then stray sentinel detection finds [NO_ANSWER] in the remaining text
    assert result.stray_sentinel is True


@pytest.mark.asyncio
async def test_mid_answer_sentinel() -> None:
    """Test mid-answer sentinel: partial answer, sentinel, then caveat."""
    reply = "Partial claim [1].\n\n[NO_ANSWER] The excerpts do not state X."
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What is X?",
        context=context,
    )

    # Should NOT be a refusal (sentinel is not leading)
    assert result.refusal is False
    # Should mark stray sentinel
    assert result.stray_sentinel is True
    # Text should keep both claim and caveat, no sentinel
    assert "Partial claim [1]." in result.text
    assert "The excerpts do not state X." in result.text
    assert NO_ANSWER_SENTINEL not in result.text


@pytest.mark.asyncio
async def test_inline_mid_sentence_sentinel() -> None:
    """Test sentinel inline mid-sentence."""
    reply = "maintaining them [NO_ANSWER] The excerpts do not state Y."
    llm = StubLLM([reply])
    context = BuiltContext(text="[1] content", chunks=[], token_count=5)

    result = await synthesize_draft(
        llm,
        question="What?",
        context=context,
    )

    assert result.refusal is False
    assert result.stray_sentinel is True
    # Text should be joined with single space
    assert result.text == "maintaining them The excerpts do not state Y."
    assert NO_ANSWER_SENTINEL not in result.text
