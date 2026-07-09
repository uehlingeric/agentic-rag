"""Tests for LLM-as-judge scoring of pipeline answers (ADR-006)."""

from __future__ import annotations

import pytest

from agentic_rag.evals.judge import (
    JudgeParseError,
    format_excerpts,
    judge_answer,
    judge_provider_for,
)
from agentic_rag.pipeline.base import CitedChunk
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import Completion, Message, Role, Usage
from agentic_rag.retrieval.base import ChunkRecord


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
        """Not implemented for judge tests."""
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


# Offline tests (all pass without external providers)


async def test_judge_happy_path() -> None:
    """Happy path: bare JSON object reply parses correctly."""
    json_reply = """{
        "faithfulness": {"score": 5, "justification": "All claims directly supported."},
        "relevance": {"score": 4, "justification": "Addresses question with minor gaps."},
        "citation_accuracy": {"score": 5, "justification": "All citations are accurate."}
    }"""
    llm = StubLLM([json_reply], name="test-judge")
    chunk = make_chunk("c1", text="Account management text here.")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="What is account management?",
        answer_text="The organization manages accounts [1].",
        cited=cited,
    )

    assert result.faithfulness.score == 5
    assert result.faithfulness.justification == "All claims directly supported."
    assert result.relevance.score == 4
    assert result.relevance.justification == "Addresses question with minor gaps."
    assert result.citation_accuracy.score == 5
    assert result.citation_accuracy.justification == "All citations are accurate."
    assert result.judge_provider == "test-judge"
    assert result.judge_model == "stub-model"
    assert result.prompt_id == load_prompt("judge").id  # latest version by default
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd == 0.001


async def test_judge_json_in_code_fences() -> None:
    """JSON wrapped in code fences and prose is still parsed."""
    reply = """Some explanation first.
```json
{
    "faithfulness": {"score": 3, "justification": "Most claims supported."},
    "relevance": {"score": 3, "justification": "Partially addresses topic."},
    "citation_accuracy": {"score": 3, "justification": "Some markers are off."}
}
```
Some explanation after."""
    llm = StubLLM([reply])
    chunk = make_chunk("c2")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 3
    assert result.relevance.score == 3
    assert result.citation_accuracy.score == 3


async def test_judge_repair_on_first_malformed() -> None:
    """First reply malformed, second valid; usage sums both calls."""
    bad_reply = "This is not JSON at all."
    good_reply = """{
        "faithfulness": {"score": 2, "justification": "Several claims lack support."},
        "relevance": {"score": 2, "justification": "Mostly misses the point."},
        "citation_accuracy": {"score": 2, "justification": "Most markers inaccurate."}
    }"""
    llm = StubLLM([bad_reply, good_reply])
    chunk = make_chunk("c3")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
        max_parse_retries=2,
    )

    assert result.faithfulness.score == 2
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


async def test_judge_all_malformed_raises() -> None:
    """All replies malformed with max_parse_retries=2 raises JudgeParseError."""
    bad1 = "not json"
    bad2 = "{incomplete"
    bad3 = '{"no": "dimensions"}'
    llm = StubLLM([bad1, bad2, bad3])
    chunk = make_chunk("c4")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    with pytest.raises(JudgeParseError) as exc_info:
        await judge_answer(
            llm,
            question="Q",
            answer_text="A [1]",
            cited=cited,
            max_parse_retries=2,
        )

    # Should have made exactly 3 calls (1 initial + 2 retries)
    assert len(llm.recorded_messages) == 3
    assert "after 3 attempts" in str(exc_info.value)


async def test_judge_score_not_int() -> None:
    """Validation: score is not an integer triggers repair."""
    bad = """{
        "faithfulness": {"score": "not_an_int", "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 1, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c5")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 1
    assert len(llm.recorded_messages) == 2


async def test_judge_score_zero() -> None:
    """Validation: score 0 (out of 1-5 range) triggers repair."""
    bad = """{
        "faithfulness": {"score": 0, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 1, "justification": "Really bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c6")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 1
    assert len(llm.recorded_messages) == 2


async def test_judge_score_six() -> None:
    """Validation: score 6 (out of 1-5 range) triggers repair."""
    bad = """{
        "faithfulness": {"score": 6, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 5, "justification": "Excellent."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c7")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 5
    assert len(llm.recorded_messages) == 2


async def test_judge_score_bool() -> None:
    """Validation: score is bool triggers repair."""
    bad = """{
        "faithfulness": {"score": true, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 1, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c8")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 1
    assert len(llm.recorded_messages) == 2


async def test_judge_score_float() -> None:
    """Validation: score is float (not int) triggers repair."""
    bad = """{
        "faithfulness": {"score": 4.0, "justification": "Bad."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 4, "justification": "Good."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c9")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 4
    assert len(llm.recorded_messages) == 2


async def test_judge_missing_justification() -> None:
    """Validation: missing justification triggers repair."""
    bad = """{
        "faithfulness": {"score": 5},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 5, "justification": "Perfect."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c10")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 5
    assert result.faithfulness.justification == "Perfect."


async def test_judge_empty_justification() -> None:
    """Validation: empty/whitespace-only justification triggers repair."""
    bad = """{
        "faithfulness": {"score": 5, "justification": "   "},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 5, "justification": "All claims supported."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c11")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.justification == "All claims supported."


async def test_judge_missing_dimension() -> None:
    """Validation: missing dimension key triggers repair."""
    bad = """{
        "faithfulness": {"score": 5, "justification": "OK."},
        "relevance": {"score": 3, "justification": "OK."}
    }"""
    good = """{
        "faithfulness": {"score": 5, "justification": "OK."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c12")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.citation_accuracy.score == 3


async def test_judge_top_level_array() -> None:
    """Validation: top-level array instead of object triggers repair."""
    bad = """[
        {"score": 5, "justification": "OK."},
        {"score": 3, "justification": "OK."}
    ]"""
    good = """{
        "faithfulness": {"score": 5, "justification": "OK."},
        "relevance": {"score": 3, "justification": "OK."},
        "citation_accuracy": {"score": 3, "justification": "OK."}
    }"""
    llm = StubLLM([bad, good])
    chunk = make_chunk("c13")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="Q",
        answer_text="A [1]",
        cited=cited,
    )

    assert result.faithfulness.score == 5


async def test_format_excerpts_empty() -> None:
    """format_excerpts: empty sequence returns standard "(none...)" string."""
    result = format_excerpts([])
    assert result == "(none — the answer cited no excerpts)"


async def test_format_excerpts_single() -> None:
    """format_excerpts: single chunk renders in correct format."""
    chunk = make_chunk("c1", heading="AC-2 ACCOUNT MANAGEMENT", text="Account text here.")
    cited = [CitedChunk(marker=1, chunk=chunk)]
    result = format_excerpts(cited)

    assert "[1] sp800-53r5 §AC-2 — AC-2 ACCOUNT MANAGEMENT" in result
    assert "Account text here." in result


async def test_format_excerpts_multiple() -> None:
    """format_excerpts: two chunks render separated by blank lines."""
    chunk1 = make_chunk("c1", section_id="AC-2", heading="Account Management", text="Text 1.")
    chunk2 = make_chunk(
        "c2",
        doc_id="at-doc",
        section_id="AT-2",
        heading="Training",
        text="Text 2.",
    )
    cited = [CitedChunk(marker=1, chunk=chunk1), CitedChunk(marker=2, chunk=chunk2)]
    result = format_excerpts(cited)

    lines = result.split("\n\n")
    assert len(lines) == 2
    assert "[1]" in lines[0] and "Account Management" in lines[0]
    assert "[2]" in lines[1] and "Training" in lines[1]
    assert "Text 1." in lines[0]
    assert "Text 2." in lines[1]


def test_judge_provider_for_picks_first_non_generation() -> None:
    """judge_provider_for: picks first judge != generation provider."""
    result = judge_provider_for("ollama", ["anthropic", "google", "ollama"])
    assert result == "anthropic"


def test_judge_provider_for_skips_same() -> None:
    """judge_provider_for: skips if first is same as generation."""
    result = judge_provider_for("anthropic", ["anthropic", "google"])
    assert result == "google"


def test_judge_provider_for_no_available() -> None:
    """judge_provider_for: raises ValueError if only option is generation provider."""
    with pytest.raises(ValueError) as exc_info:
        judge_provider_for("anthropic", ["anthropic"])

    assert "no judge available" in str(exc_info.value)


def test_judge_provider_for_empty() -> None:
    """judge_provider_for: raises ValueError on empty preference list."""
    with pytest.raises(ValueError):
        judge_provider_for("anthropic", [])


async def test_judge_prompt_version_pinning() -> None:
    """judge_answer: version=1 pins to judge.v1 prompt."""
    json_reply = """{
        "faithfulness": {"score": 5, "justification": "OK."},
        "relevance": {"score": 5, "justification": "OK."},
        "citation_accuracy": {"score": 5, "justification": "OK."}
    }"""
    llm = StubLLM([json_reply])
    chunk = make_chunk("c1")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    result = await judge_answer(
        llm,
        question="What is X?",
        answer_text="X is Y [1].",
        cited=cited,
        prompt_version=1,
    )

    assert result.prompt_id == "judge.v1"


async def test_judge_rendered_prompt_contains_question_answer_excerpts() -> None:
    """Rendered prompt (first user message) contains question, answer, and excerpts."""
    json_reply = """{
        "faithfulness": {"score": 5, "justification": "OK."},
        "relevance": {"score": 5, "justification": "OK."},
        "citation_accuracy": {"score": 5, "justification": "OK."}
    }"""
    llm = StubLLM([json_reply])
    chunk = make_chunk("c1", text="Excerpt text here.")
    cited = [CitedChunk(marker=1, chunk=chunk)]

    question = "What is account management?"
    answer = "Account management is important [1]."

    await judge_answer(
        llm,
        question=question,
        answer_text=answer,
        cited=cited,
    )

    # First call's first message should have all three
    first_user_msg = llm.recorded_messages[0][0].content
    assert question in first_user_msg
    assert answer in first_user_msg
    assert "Excerpt text here." in first_user_msg


# Live tests (marked to run manually with real providers)


@pytest.mark.live
async def test_judge_live_correct_answer() -> None:
    """Live: correct answer citing relevant excerpt scores high on all dimensions."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    judge_provider_name = settings.judge.providers[0]
    judge = get_llm_provider(judge_provider_name, settings)

    # NIST SP 800-53 Rev. 5: AC-2 Account Management
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

    cited = [CitedChunk(marker=1, chunk=ac2)]
    question = "How does the organization manage system accounts?"
    answer = (
        "Organizations manage system accounts by creating, modifying, and removing accounts "
        "as needed [1], with accounts scoped to authorized individuals and services."
    )

    result = await judge_answer(
        judge,
        question=question,
        answer_text=answer,
        cited=cited,
    )

    # Correct, on-topic, well-cited answer should score >= 4 on all dimensions
    assert result.faithfulness.score >= 4, f"Faithfulness too low: {result.faithfulness}"
    assert result.relevance.score >= 4, f"Relevance too low: {result.relevance}"
    assert result.citation_accuracy.score >= 4, (
        f"Citation accuracy too low: {result.citation_accuracy}"
    )


@pytest.mark.live
async def test_judge_live_wrong_excerpt_citation() -> None:
    """Live: answer citing wrong excerpt scores low on citation_accuracy."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    judge_provider_name = settings.judge.providers[0]
    judge = get_llm_provider(judge_provider_name, settings)

    # Two separate excerpts
    ac2 = make_chunk(
        "ac2-chunk",
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

    # The claim comes from AC-2's text, but the marker points at the AT-2 excerpt:
    # a misattributed citation, isolated from relevance (the answer is on-topic).
    cited = [
        CitedChunk(marker=1, chunk=ac2),
        CitedChunk(marker=2, chunk=at2),
    ]
    question = "How does the organization manage system accounts?"
    answer = (
        "Account management includes the creation, activation, modification, disablement, "
        "and removal of system accounts [2]."
    )

    result = await judge_answer(
        judge,
        question=question,
        answer_text=answer,
        cited=cited,
    )

    # Citation is wrong (AT-2 doesn't explain account management)
    assert result.citation_accuracy.score <= 2, (
        f"Citation accuracy too high for wrong excerpt: {result.citation_accuracy}"
    )


@pytest.mark.live
async def test_judge_live_unsupported_claim() -> None:
    """Live: answer claiming facts not in excerpts scores low on faithfulness."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    judge_provider_name = settings.judge.providers[0]
    judge = get_llm_provider(judge_provider_name, settings)

    ac2 = make_chunk(
        "ac2-chunk",
        section_id="AC-2",
        heading="AC-2 ACCOUNT MANAGEMENT",
        text=(
            "Organizations manage system accounts. Account management includes the creation, "
            "activation, modification, disablement, and removal of system accounts."
        ),
    )

    cited = [CitedChunk(marker=1, chunk=ac2)]
    question = "What are the account password requirements?"
    answer = (
        "Passwords must be rotated every 90 days and contain uppercase, lowercase, "
        "numbers, and special characters [1]."
    )

    result = await judge_answer(
        judge,
        question=question,
        answer_text=answer,
        cited=cited,
    )

    # Claims in answer are not in the excerpt
    assert result.faithfulness.score <= 2, (
        f"Faithfulness too high for unsupported claims: {result.faithfulness}"
    )


@pytest.mark.live
async def test_judge_live_off_topic_answer() -> None:
    """Live: fluent but off-topic answer scores low on relevance."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    judge_provider_name = settings.judge.providers[0]
    judge = get_llm_provider(judge_provider_name, settings)

    ac2 = make_chunk(
        "ac2-chunk",
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

    cited = [
        CitedChunk(marker=1, chunk=ac2),
        CitedChunk(marker=2, chunk=at2),
    ]
    question = "How does the organization manage system accounts?"
    answer = (
        "Personnel must receive comprehensive security and privacy training before "
        "accessing the system [2]. This ensures they understand the organization's policies."
    )

    result = await judge_answer(
        judge,
        question=question,
        answer_text=answer,
        cited=cited,
    )

    # Answer is about training, not account management (wrong topic)
    assert result.relevance.score <= 2, (
        f"Relevance too high for off-topic answer: {result.relevance}"
    )
