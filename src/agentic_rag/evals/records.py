"""JSONL row schema for generation-eval results.

One row per (example, config) carries everything needed to aggregate,
re-judge, and audit a run without re-running generation: the answer text, the
chunk ids it cited, per-stage latency, token usage, and the judge verdict.
Excerpt text is not stored — cited chunk ids resolve against the corpus
(``data/corpus/chunks.jsonl``), whose build is byte-reproducible, keeping
result files small. Readers (aggregation, report, re-judge) consume the raw
JSON dicts; only writers construct these dataclasses.

Field inventory (per-row): example_id, dataset_version, provider, model, mode,
rerank, pipeline, synthesis_prompt, answer_text, refusal, cited, invalid_citations,
n_context, gen_usage, latency_s, judge, agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.evals.judge import JudgeScores
from agentic_rag.providers.base import Usage


@dataclass(frozen=True, slots=True)
class AgentMeta:
    """Agentic-loop row metadata. ``caveat`` means the revision cap was hit
    while the critic still said revise."""

    plan_kind: str  # "direct" | "multi_hop"
    sub_queries: list[str]
    revisions: int
    caveat: bool


@dataclass(frozen=True, slots=True)
class CitedRef:
    """One resolved inline citation, by reference: marker ``[n]`` and chunk identity."""

    marker: int
    chunk_id: str
    doc: str
    section: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One (example, config) result row.

    ``judge`` is None when the answer was a refusal (refusals are scored as
    refusal correctness, not on the rubrics) or when judging is deferred.
    ``latency_s`` maps pipeline stage name -> wall-clock seconds.
    ``agent`` is None for vanilla rows.
    Agentic rows' latency_s stages are planner/retrieve/synthesize/critic
    (rerank time folds into retrieve inside gather); vanilla rows keep
    retrieve/rerank/synthesize.
    """

    example_id: str
    dataset_version: str
    provider: str
    model: str
    mode: str
    rerank: str
    pipeline: str
    synthesis_prompt: str
    answer_text: str
    refusal: bool
    cited: list[CitedRef]
    invalid_citations: list[int]
    n_context: int
    gen_usage: Usage
    latency_s: dict[str, float]
    judge: JudgeScores | None
    agent: AgentMeta | None
    guardrails: bool = False


def usage_to_json(usage: Usage) -> dict[str, object]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def judge_to_json(scores: JudgeScores) -> dict[str, object]:
    return {
        "judge_provider": scores.judge_provider,
        "judge_model": scores.judge_model,
        "prompt_id": scores.prompt_id,
        "faithfulness": {
            "score": scores.faithfulness.score,
            "justification": scores.faithfulness.justification,
        },
        "relevance": {
            "score": scores.relevance.score,
            "justification": scores.relevance.justification,
        },
        "citation_accuracy": {
            "score": scores.citation_accuracy.score,
            "justification": scores.citation_accuracy.justification,
        },
        "usage": usage_to_json(scores.usage),
    }


def record_to_json(record: GenerationRecord) -> dict[str, object]:
    return {
        "example_id": record.example_id,
        "dataset_version": record.dataset_version,
        "provider": record.provider,
        "model": record.model,
        "mode": record.mode,
        "rerank": record.rerank,
        "pipeline": record.pipeline,
        "synthesis_prompt": record.synthesis_prompt,
        "answer_text": record.answer_text,
        "refusal": record.refusal,
        "cited": [
            {"marker": c.marker, "chunk_id": c.chunk_id, "doc": c.doc, "section": c.section}
            for c in record.cited
        ],
        "invalid_citations": record.invalid_citations,
        "n_context": record.n_context,
        "gen_usage": usage_to_json(record.gen_usage),
        "latency_s": record.latency_s,
        "judge": judge_to_json(record.judge) if record.judge is not None else None,
        "guardrails": record.guardrails,
        "agent": {
            "plan_kind": record.agent.plan_kind,
            "sub_queries": record.agent.sub_queries,
            "revisions": record.agent.revisions,
            "caveat": record.agent.caveat,
        }
        if record.agent is not None
        else None,
    }
