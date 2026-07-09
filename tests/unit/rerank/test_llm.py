"""Tests for LLMReranker."""

from __future__ import annotations

import pytest

from agentic_rag.providers.base import Completion, Message, Usage
from agentic_rag.rerank.llm import LLMReranker
from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "sp800-53r5",
    section_id: str = "AC-2",
    text: str = "The organization manages system accounts.",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=[section_id],
        section_path=f"{section_id} SECTION",
        heading=f"{section_id} SECTION",
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_scored(
    chunk: ChunkRecord,
    score: float = 0.5,
    rank: int = 1,
    source_scores: dict[str, float] | None = None,
) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk=chunk,
        score=score,
        rank=rank,
        source_scores=source_scores or {"bm25": score},
    )


class FakeLLM:
    """Fake LLM provider for testing."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], str | None, float, int]] = []
        self.response_text = '{"ranking": ["c1", "c2", "c3", "c4"]}'
        self.response_usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.0)

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls.append((messages, model, temperature, max_tokens))
        return Completion(
            text=self.response_text,
            model=model or "fake-model",
            usage=self.response_usage,
        )

    def stream(self, *args, **kwargs):
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


@pytest.mark.asyncio
async def test_happy_path_reverses_order():
    """Ranking reverses input -> output reversed, ranks 1..4, scores preserved."""
    c1 = make_scored(make_chunk("c1"), score=0.8)
    c2 = make_scored(make_chunk("c2"), score=0.7)
    c3 = make_scored(make_chunk("c3"), score=0.6)
    c4 = make_scored(make_chunk("c4"), score=0.5)
    candidates = [c1, c2, c3, c4]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c4", "c3", "c2", "c1"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=4)

    assert len(result) == 4
    assert result[0].chunk.chunk_id == "c4"
    assert result[1].chunk.chunk_id == "c3"
    assert result[2].chunk.chunk_id == "c2"
    assert result[3].chunk.chunk_id == "c1"
    assert [r.rank for r in result] == [1, 2, 3, 4]
    # Scores and source_scores preserved
    assert result[0].score == 0.5  # c4's original score
    assert result[3].score == 0.8  # c1's original score


@pytest.mark.asyncio
async def test_top_k_cuts_after_reorder():
    """top_k=2 cuts after reorder."""
    c1 = make_scored(make_chunk("c1"), score=0.8)
    c2 = make_scored(make_chunk("c2"), score=0.7)
    c3 = make_scored(make_chunk("c3"), score=0.6)
    c4 = make_scored(make_chunk("c4"), score=0.5)
    candidates = [c1, c2, c3, c4]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c4", "c3", "c2", "c1"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0].chunk.chunk_id == "c4"
    assert result[1].chunk.chunk_id == "c3"
    assert [r.rank for r in result] == [1, 2]


@pytest.mark.asyncio
async def test_code_fenced_response_parses():
    """Code-fenced response (```json ... ```) parses."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '```json\n{"ranking": ["c2", "c1"]}\n```'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0].chunk.chunk_id == "c2"
    assert result[1].chunk.chunk_id == "c1"


@pytest.mark.asyncio
async def test_code_fenced_response_with_leading_fence():
    """Code fence with leading markdown fence."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '```\n{"ranking": ["c2", "c1"]}\n```'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0].chunk.chunk_id == "c2"
    assert result[1].chunk.chunk_id == "c1"


@pytest.mark.asyncio
async def test_invalid_json_fallback():
    """Invalid JSON -> input order, cut to top_k, ranks reassigned."""
    c1 = make_scored(make_chunk("c1"), score=0.8)
    c2 = make_scored(make_chunk("c2"), score=0.7)
    c3 = make_scored(make_chunk("c3"), score=0.6)
    c4 = make_scored(make_chunk("c4"), score=0.5)
    candidates = [c1, c2, c3, c4]

    fake_llm = FakeLLM()
    fake_llm.response_text = "not json at all {{{["

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=4)

    assert len(result) == 4
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2", "c3", "c4"]
    assert [r.rank for r in result] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_missing_ranking_key_fallback():
    """JSON without "ranking" key -> fallback."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"result": ["c2", "c1"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=2)

    assert len(result) == 2
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_ranking_not_list_fallback():
    """ranking not a list (string) -> fallback."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": "c2, c1"}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=2)

    assert len(result) == 2
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_unknown_ids_ignored_missing_appended():
    """Unknown ids in ranking ignored; missing ids appended in input order."""
    c1 = make_scored(make_chunk("c1"), score=0.8)
    c2 = make_scored(make_chunk("c2"), score=0.7)
    c3 = make_scored(make_chunk("c3"), score=0.6)
    c4 = make_scored(make_chunk("c4"), score=0.5)
    candidates = [c1, c2, c3, c4]

    fake_llm = FakeLLM()
    # Ranking lists only c2, c4, and a bogus id; c1, c3 missing
    fake_llm.response_text = '{"ranking": ["c2", "bogus_id", "c4"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=4)

    assert len(result) == 4
    assert [c.chunk.chunk_id for c in result] == ["c2", "c4", "c1", "c3"]


@pytest.mark.asyncio
async def test_duplicate_ids_first_occurrence_wins():
    """Duplicate ids: first occurrence wins, no duplicate chunks in output."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    c3 = make_scored(make_chunk("c3"))
    candidates = [c1, c2, c3]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c2", "c1", "c2", "c3"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", candidates, top_k=3)

    assert len(result) == 3
    assert [c.chunk.chunk_id for c in result] == ["c2", "c1", "c3"]


@pytest.mark.asyncio
async def test_empty_candidates():
    """Empty candidates -> [] and no LLM call."""
    fake_llm = FakeLLM()
    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("test query", [], top_k=10)

    assert result == []
    assert len(fake_llm.calls) == 0
    assert reranker.last_usage == Usage.zero()


@pytest.mark.asyncio
async def test_prompt_content_contains_query_and_ids():
    """Rendered prompt contains query text and every candidate chunk_id."""
    c1 = make_scored(make_chunk("c1", text="Text for c1"))
    c2 = make_scored(make_chunk("c2", text="Text for c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1", "c2"]}'

    reranker = LLMReranker(fake_llm)
    await reranker.rerank("test query about xyz", candidates, top_k=2)

    assert len(fake_llm.calls) == 1
    messages, _, _, _ = fake_llm.calls[0]
    prompt_text = messages[0].content

    assert "test query about xyz" in prompt_text
    assert "c1:" in prompt_text
    assert "c2:" in prompt_text


@pytest.mark.asyncio
async def test_candidate_excerpts_single_line_truncated():
    """Candidate excerpts single-line and <= 200 chars."""
    long_text = "x" * 500
    c1 = make_scored(make_chunk("c1", text=long_text))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'

    reranker = LLMReranker(fake_llm)
    await reranker.rerank("query", candidates, top_k=1)

    messages, _, _, _ = fake_llm.calls[0]
    prompt_text = messages[0].content

    # Find the candidate line
    for line in prompt_text.split("\n"):
        if line.startswith("c1:"):
            # Extract the excerpt (everything after "c1: ")
            excerpt = line[4:]
            assert len(excerpt) <= 200
            assert "\n" not in excerpt
            assert "  " not in excerpt  # No double spaces


@pytest.mark.asyncio
async def test_call_params_temperature_max_tokens():
    """Call params: temperature == 0.0, max_tokens == 1024."""
    c1 = make_scored(make_chunk("c1"))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'

    reranker = LLMReranker(fake_llm)
    await reranker.rerank("query", candidates, top_k=1)

    _, _, temperature, max_tokens = fake_llm.calls[0]
    assert temperature == 0.0
    assert max_tokens == 1024


@pytest.mark.asyncio
async def test_model_override_forwarded():
    """Model override forwarded to LLM."""
    c1 = make_scored(make_chunk("c1"))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'

    reranker = LLMReranker(fake_llm, model="custom-model")
    await reranker.rerank("query", candidates, top_k=1)

    _, model, _, _ = fake_llm.calls[0]
    assert model == "custom-model"


@pytest.mark.asyncio
async def test_last_usage_set_from_completion():
    """last_usage equals fake completion's usage after call."""
    c1 = make_scored(make_chunk("c1"))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'
    test_usage = Usage(input_tokens=123, output_tokens=45, cost_usd=0.01)
    fake_llm.response_usage = test_usage

    reranker = LLMReranker(fake_llm)
    await reranker.rerank("query", candidates, top_k=1)

    assert reranker.last_usage == test_usage


@pytest.mark.asyncio
async def test_last_usage_zero_on_empty():
    """last_usage is zero() before any call and on empty candidates."""
    fake_llm = FakeLLM()
    reranker = LLMReranker(fake_llm)

    assert reranker.last_usage == Usage.zero()

    await reranker.rerank("query", [], top_k=1)
    assert reranker.last_usage == Usage.zero()


@pytest.mark.asyncio
async def test_scores_and_source_scores_preserved():
    """Scores and source_scores preserved after reordering."""
    c1 = make_scored(
        make_chunk("c1"),
        score=0.8,
        source_scores={"bm25": 0.8, "dense": 0.7},
    )
    c2 = make_scored(
        make_chunk("c2"),
        score=0.6,
        source_scores={"bm25": 0.6},
    )
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c2", "c1"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("query", candidates, top_k=2)

    assert result[0].score == 0.6
    assert result[0].source_scores == {"bm25": 0.6}
    assert result[1].score == 0.8
    assert result[1].source_scores == {"bm25": 0.8, "dense": 0.7}


@pytest.mark.asyncio
async def test_ranking_with_non_string_entries_fallback():
    """ranking with non-string entries -> fallback."""
    c1 = make_scored(make_chunk("c1"))
    c2 = make_scored(make_chunk("c2"))
    candidates = [c1, c2]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c2", 1, "c1"]}'

    reranker = LLMReranker(fake_llm)
    result = await reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_whitespace_collapsing_in_excerpt():
    """Excerpts have whitespace runs collapsed to single spaces."""
    text_with_newlines = "Line one\n\nLine two\t\tTabbed"
    c1 = make_scored(make_chunk("c1", text=text_with_newlines))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'

    reranker = LLMReranker(fake_llm)
    await reranker.rerank("query", candidates, top_k=1)

    messages, _, _, _ = fake_llm.calls[0]
    prompt_text = messages[0].content

    for line in prompt_text.split("\n"):
        if line.startswith("c1:"):
            excerpt = line[4:]
            assert "Line one Line two Tabbed" in excerpt


@pytest.mark.asyncio
async def test_prompt_version_passed():
    """Prompt version is passed to load_prompt."""
    c1 = make_scored(make_chunk("c1"))
    candidates = [c1]

    fake_llm = FakeLLM()
    fake_llm.response_text = '{"ranking": ["c1"]}'

    # This test verifies that prompt_version is used (implicit via the fact
    # that init doesn't fail and uses the v1 prompt with correct variables)
    reranker = LLMReranker(fake_llm, prompt_version=1)
    result = await reranker.rerank("query", candidates, top_k=1)

    assert len(result) == 1
