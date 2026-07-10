"""Tests for query planner classification and decomposition (ADR-007)."""

from __future__ import annotations

from agentic_rag.agent.planner import PlanResult, plan_query
from agentic_rag.agent.state import PlanKind
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import Completion, Message, Role, Usage


class StubLLM:
    """Stub LLM provider for testing: pops replies from a list and records calls."""

    def __init__(self, replies: list[str], name: str = "stub-planner") -> None:
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
        """Not implemented for planner tests."""
        raise NotImplementedError("stream() not implemented in StubLLM")

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split())


async def test_direct_reply() -> None:
    """Test 1: Direct reply → DIRECT plan, sub_queries == (original question,), fallback False."""
    llm = StubLLM(['{"classification": "direct"}'])
    question = "What is control AC-2?"

    result = await plan_query(llm, question)

    assert isinstance(result, PlanResult)
    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is False
    assert result.raw == '{"classification": "direct"}'


async def test_multi_hop_three_queries() -> None:
    """Test 2: Multi-hop with 3 sub-queries → MULTI_HOP, correct order, fallback False."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": [
            "SP 800-53 Revision 5 control AC-2 account management requirements",
            "FIPS 200 minimum security requirements",
            "Relationship between AC-2 and FIPS 200"
        ]
    }"""
    llm = StubLLM([reply])
    question = "How do account management requirements relate to security objectives?"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert len(result.plan.sub_queries) == 3
    assert (
        result.plan.sub_queries[0]
        == "SP 800-53 Revision 5 control AC-2 account management requirements"
    )
    assert result.plan.sub_queries[1] == "FIPS 200 minimum security requirements"
    assert result.plan.sub_queries[2] == "Relationship between AC-2 and FIPS 200"
    assert result.fallback is False


async def test_json_wrapped_in_prose() -> None:
    """Test 3: JSON wrapped in prose and code fences → still parses."""
    reply = """Let me analyze this question.
```json
{"classification": "direct"}
```
This appears to be straightforward."""
    llm = StubLLM([reply])
    question = "What does AC-10 require?"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is False


async def test_six_queries_max_four() -> None:
    """Test 4: Six sub-queries with max_sub_queries=4 → first 4 kept, fallback False."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": [
            "Query 1",
            "Query 2",
            "Query 3",
            "Query 4",
            "Query 5",
            "Query 6"
        ]
    }"""
    llm = StubLLM([reply])
    question = "Complex multi-part question"

    result = await plan_query(llm, question, max_sub_queries=4)

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert len(result.plan.sub_queries) == 4
    assert result.plan.sub_queries == ("Query 1", "Query 2", "Query 3", "Query 4")
    assert result.fallback is False


async def test_multi_hop_one_valid_query_fallback() -> None:
    """Test 5: Multi-hop with 1 valid sub-query → DIRECT with original question, fallback True."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": ["Only one valid query"]
    }"""
    llm = StubLLM([reply])
    question = "What is the original question?"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True


async def test_multi_hop_all_empty_fallback() -> None:
    """Test 5b: Multi-hop with all entries empty after stripping → DIRECT fallback True."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": ["   ", "", "  \t  "]
    }"""
    llm = StubLLM([reply])
    question = "Original question"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True


async def test_garbage_no_json_fallback() -> None:
    """Test 6: Garbage reply (no JSON) → DIRECT fallback True; raw preserved."""
    garbage = "This is completely invalid and has no JSON at all."
    llm = StubLLM([garbage])
    question = "Test question"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True
    assert result.raw == garbage


async def test_invalid_json_fallback() -> None:
    """Test 6b: Malformed JSON → DIRECT fallback True."""
    malformed = '{"classification": "direct"' + " incomplete"
    llm = StubLLM([malformed])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.fallback is True


async def test_unknown_classification_fallback() -> None:
    """Test 7: Unknown classification value → DIRECT fallback True."""
    reply = '{"classification": "unknown_type"}'
    llm = StubLLM([reply])
    question = "Test question"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True


async def test_sub_queries_non_string_dropped() -> None:
    """Test 8: sub_queries with non-string entries → they are dropped."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": [
            "First valid string",
            123,
            "Second valid string",
            null,
            "Third valid string",
            true,
            false
        ]
    }"""
    llm = StubLLM([reply])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert len(result.plan.sub_queries) == 3
    assert result.plan.sub_queries == (
        "First valid string",
        "Second valid string",
        "Third valid string",
    )
    assert result.fallback is False


async def test_non_string_entries_degenerate_to_fallback() -> None:
    """Test 8b: Filtering non-strings leaves <2 → DIRECT fallback True."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": [123, null, "only one string", true, false]
    }"""
    llm = StubLLM([reply])
    question = "Test question"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True


async def test_prompt_integrity() -> None:
    """Test 9: Prompt loading, rendering, and prompt_id attribution."""
    # This verifies load_prompt("planner") works and renders correctly
    prompt = load_prompt("planner")
    assert prompt.name == "planner"
    assert prompt.version == 1  # Latest/only version
    assert prompt.id == "planner.v1"

    # Verify rendering doesn't KeyError on a question
    test_question = "Does control SC-7 require boundary protection?"
    rendered = prompt.render(question=test_question)
    assert test_question in rendered
    assert "${question}" not in rendered  # Template substituted

    # Now call the planner and check prompt_id
    reply = '{"classification": "direct"}'
    llm = StubLLM([reply])

    result = await plan_query(llm, test_question)
    assert result.prompt_id == "planner.v1"


async def test_llm_called_with_correct_params() -> None:
    """Test 10: LLM was called with temperature 0.0 and single user message."""
    question = "Test question for param check"
    reply = '{"classification": "direct"}'
    llm = StubLLM([reply])

    await plan_query(
        llm,
        question,
        model="test-model",
        max_tokens=256,
        prompt_version=1,
    )

    # Check that complete() was called exactly once
    assert len(llm.recorded_temps) == 1
    assert llm.recorded_temps[0] == 0.0

    assert len(llm.recorded_messages) == 1
    messages = llm.recorded_messages[0]
    assert len(messages) == 1
    assert messages[0].role == Role.USER
    assert question in messages[0].content

    assert len(llm.recorded_max_tokens) == 1
    assert llm.recorded_max_tokens[0] == 256

    assert len(llm.recorded_models) == 1
    assert llm.recorded_models[0] == "test-model"


async def test_usage_attribution() -> None:
    """Usage from completion is correctly attributed in PlanResult."""
    llm = StubLLM(['{"classification": "direct"}'])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd == 0.001


async def test_sub_queries_not_a_list_fallback() -> None:
    """If sub_queries is not a list → DIRECT fallback True."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": "not a list"
    }"""
    llm = StubLLM([reply])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.plan.sub_queries == (question,)
    assert result.fallback is True


async def test_top_level_array_instead_of_object_fallback() -> None:
    """If reply is a JSON array instead of object → DIRECT fallback True."""
    reply = '["classification", "direct"]'
    llm = StubLLM([reply])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.fallback is True


async def test_missing_classification_fallback() -> None:
    """If classification key is missing → DIRECT fallback True."""
    reply = '{"sub_queries": ["q1", "q2"]}'
    llm = StubLLM([reply])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.DIRECT
    assert result.fallback is True


async def test_multi_hop_with_whitespace_padding() -> None:
    """Multi-hop queries with surrounding whitespace are stripped."""
    reply = """{
        "classification": "multi_hop",
        "sub_queries": [
            "  Query One  ",
            " Query Two ",
            "   Query Three   "
        ]
    }"""
    llm = StubLLM([reply])
    question = "Test"

    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert result.plan.sub_queries == ("Query One", "Query Two", "Query Three")
    assert result.fallback is False


async def test_default_max_sub_queries() -> None:
    """Default max_sub_queries is 4 (not changed by call)."""
    # Create a reply with more than 4 queries
    reply = """{
        "classification": "multi_hop",
        "sub_queries": ["q1", "q2", "q3", "q4", "q5"]
    }"""
    llm = StubLLM([reply])
    question = "Test"

    # Call without specifying max_sub_queries (should default to 4)
    result = await plan_query(llm, question)

    assert result.plan.kind == PlanKind.MULTI_HOP
    assert len(result.plan.sub_queries) == 4
    assert result.plan.sub_queries == ("q1", "q2", "q3", "q4")
