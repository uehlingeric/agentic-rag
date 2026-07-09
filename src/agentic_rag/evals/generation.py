"""Generation evaluation: end-to-end RAG answer quality with LLM-as-judge scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.evals.judge import JudgeParseError, judge_answer, judge_provider_for
from agentic_rag.evals.records import CitedRef, GenerationRecord, record_to_json
from agentic_rag.evals.retrieval import GoldenExample
from agentic_rag.providers.pricing import cost_for
from agentic_rag.providers.registry import get_embedding_provider, get_llm_provider
from agentic_rag.rerank.base import NoopReranker, Reranker
from agentic_rag.retrieval.base import RetrievalMode
from agentic_rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Cost estimation constants (in tokens)
EST_GEN_INPUT = 6500  # synthesis.max_context_tokens (6000) + 500 buffer
EST_GEN_OUTPUT = 300
EST_JUDGE_INPUT = 2500
EST_JUDGE_OUTPUT = 250


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for one generation evaluation run."""

    provider: str
    mode: str
    rerank: str

    def slug(self) -> str:
        """Return identifier slug for this config (provider--mode--rerank)."""
        return f"{self.provider}--{self.mode}--{self.rerank}"


def config_settings(base: Settings, cfg: RunConfig) -> Settings:
    """Build per-config settings by swapping provider and rerank mode."""
    return base.model_copy(
        update={
            "provider": cfg.provider,
            "rerank": base.rerank.model_copy(update={"mode": cfg.rerank}),
        }
    )


def _get_provider_model(provider: str, settings: Settings) -> str:
    """Get the model string for a provider from settings."""
    if provider == "anthropic":
        if settings.anthropic.backend == "bedrock":
            return settings.anthropic.bedrock_model or settings.anthropic.model
        return settings.anthropic.model
    elif provider == "google":
        return settings.google.model
    elif provider == "openai":
        return settings.openai.model
    elif provider == "ollama":
        return settings.ollama.model
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def run_config(
    cfg: RunConfig,
    golden: list[GoldenExample],
    settings: Settings,
    out_path: Path,
    *,
    dataset_version: str,
    concurrency: int = 4,
    do_judge: bool = True,
    _pipeline_factory: Callable[[Settings], Any] | None = None,  # For testing
) -> None:
    """Run generation evaluation for one config, writing results to out_path.

    Resumes from existing results; skips already-evaluated examples. Runs
    synthesis and optionally judging, bounded by concurrency limit.

    Args:
        cfg: RunConfig with provider, mode, rerank.
        golden: List of GoldenExample to evaluate.
        settings: Base application settings.
        out_path: Path to write JSONL results (created if needed).
        dataset_version: Version tag recorded on every row (the dataset file stem).
        concurrency: Max concurrent synthesis/judge calls.
        do_judge: If False, write judge=None for all rows.
        _pipeline_factory: Callable(settings) -> RAGPipeline (for testing).
    """
    # Resume: load existing example_ids and skip them
    existing_ids = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing_ids.add(row["example_id"])

    # Build per-config settings
    cfg_settings = config_settings(settings, cfg)

    # Build pipeline once
    if _pipeline_factory is not None:
        pipeline = _pipeline_factory(cfg_settings)
    else:
        from agentic_rag.pipeline.pipeline import RAGPipeline
        from agentic_rag.rerank.cross_encoder import CrossEncoderReranker
        from agentic_rag.rerank.llm import LLMReranker

        llm = get_llm_provider(cfg.provider, cfg_settings)
        embedder = get_embedding_provider(cfg_settings.embedding.provider, cfg_settings)
        retriever = Retriever.load(
            cfg_settings.data_dir / "index",
            embedder,
            rrf_k=cfg_settings.retrieval.rrf_k,
            candidate_pool=cfg_settings.retrieval.candidate_pool,
        )
        reranker: Reranker
        if cfg.rerank == "none":
            reranker = NoopReranker()
        elif cfg.rerank == "llm":
            reranker = LLMReranker(llm, model=cfg_settings.rerank.model)
        elif cfg.rerank == "cross-encoder":
            reranker = CrossEncoderReranker(model=cfg_settings.rerank.model)
        else:
            raise ValueError(f"Unknown rerank mode: {cfg.rerank}")
        pipeline = RAGPipeline(retriever, reranker, llm, cfg_settings)

    # Build judge LLM if needed
    judge_llm = None
    if do_judge:
        judge_provider = judge_provider_for(cfg.provider, cfg_settings.judge.providers)
        judge_llm = get_llm_provider(judge_provider, cfg_settings)

    # Get synthesis prompt id
    from agentic_rag.prompts import load_prompt

    synthesis_prompt = load_prompt("synthesis", version=None)
    synthesis_prompt_id = synthesis_prompt.id

    # Get provider model
    provider_model = _get_provider_model(cfg.provider, cfg_settings)

    # Write results with async lock
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    retrieval_mode = RetrievalMode(cfg.mode)

    async def process_example(example: GoldenExample) -> None:
        """Process one example: ask, judge, write result."""
        if example.id in existing_ids:
            return

        async with semaphore:
            try:
                # Run synthesis
                answer = await pipeline.ask(example.question, mode=retrieval_mode)

                # Build cited refs
                cited_refs = [
                    CitedRef(
                        marker=c.marker,
                        chunk_id=c.chunk.chunk_id,
                        doc=c.chunk.doc_id,
                        section=c.chunk.section_id,
                    )
                    for c in answer.citations
                ]

                # Build latency dict
                latency_s = {t.stage: t.seconds for t in answer.timings}
                latency_s["total"] = sum(t.seconds for t in answer.timings)

                # Judge unless refusal or do_judge=False
                judge_scores = None
                if not answer.refusal and do_judge and judge_llm is not None:
                    try:
                        judge_scores = await judge_answer(
                            judge_llm,
                            question=example.question,
                            answer_text=answer.text,
                            cited=answer.citations,
                            prompt_version=cfg_settings.judge.prompt_version,
                            max_tokens=cfg_settings.judge.max_tokens,
                            max_parse_retries=cfg_settings.judge.max_parse_retries,
                        )
                    except JudgeParseError as exc:
                        logger.warning(
                            f"Judge parse failed for {example.id}: {exc}; scoring as None"
                        )
                        judge_scores = None

                # Build record
                record = GenerationRecord(
                    example_id=example.id,
                    dataset_version=dataset_version,
                    provider=cfg.provider,
                    model=provider_model,
                    mode=cfg.mode,
                    rerank=cfg.rerank,
                    synthesis_prompt=synthesis_prompt_id,
                    answer_text=answer.text,
                    refusal=answer.refusal,
                    cited=cited_refs,
                    invalid_citations=answer.invalid_citations,
                    n_context=len(answer.context),
                    gen_usage=answer.usage,
                    latency_s=latency_s,
                    judge=judge_scores,
                )

                # Write row
                async with write_lock:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record_to_json(record)) + "\n")

            except Exception as exc:
                logger.error(f"Error processing {example.id}: {exc}", exc_info=True)
                raise

    # Process all examples concurrently
    tasks = [process_example(ex) for ex in golden if ex.id not in existing_ids]
    await asyncio.gather(*tasks)


def estimate_cost(
    configs: list[RunConfig], n_examples: int, settings: Settings
) -> list[tuple[RunConfig, float | None]]:
    """Estimate cost for each config: generation + judging (if judge available).

    Uses heuristics:
    - Generation: EST_GEN_INPUT + EST_GEN_OUTPUT tokens per example
    - Judging: EST_JUDGE_INPUT + EST_JUDGE_OUTPUT tokens per answered example

    Ollama always costs 0.0. Unknown models return None (unpriceable).
    """
    results: list[tuple[RunConfig, float | None]] = []
    for cfg in configs:
        cfg_settings = config_settings(settings, cfg)

        # Get generation provider model
        gen_model = _get_provider_model(cfg.provider, cfg_settings)
        gen_cost = cost_for(cfg.provider, gen_model, EST_GEN_INPUT, EST_GEN_OUTPUT)
        gen_total = gen_cost * n_examples if gen_cost is not None else None
        if gen_total is None:  # Unpriceable generation model: whole config is unpriceable
            results.append((cfg, None))
            continue

        try:
            judge_provider = judge_provider_for(cfg.provider, cfg_settings.judge.providers)
            judge_model = _get_provider_model(judge_provider, cfg_settings)
            judge_cost = cost_for(judge_provider, judge_model, EST_JUDGE_INPUT, EST_JUDGE_OUTPUT)
            judge_total = judge_cost * n_examples if judge_cost is not None else None
        except ValueError:
            judge_total = None

        # Sum generation + judging
        total = None if judge_total is None else gen_total + judge_total

        results.append((cfg, total))

    return results


def summarize(run_dir: Path, golden: list[GoldenExample]) -> dict[str, object]:
    """Aggregate results from all JSONL files in run_dir, return summary.json structure.

    Groups rows by (provider, model, mode, rerank) and computes aggregates:
    - n_items: total rows in group
    - n_judged: rows with judge != None and not refusal
    - n_refusals: rows with refusal=True
    - n_judge_failures: non-refusal rows with judge=None
    - refusal_correct_rate: refused unanswerable / total unanswerable
    - false_refusal_rate: refused answerable / total answerable
    - scores: mean of faithfulness/relevance/citation_accuracy over judged rows
    - latency_s: mean, p50, p95 of row's total latency
    - gen_tokens: sum of input/output tokens
    - gen_cost_usd: sum of costs (None if any row is None)
    - judge_cost_usd: same for judge usage
    """
    # Load all rows from JSONL files in run_dir; rejudge outputs live beside the
    # originals and carry the same config keys, so they must not double-count here
    all_rows: list[dict[str, Any]] = []
    for jsonl_file in sorted(run_dir.glob("*.jsonl")):
        if ".rejudge" in jsonl_file.name:
            continue
        with jsonl_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))

    # Build golden lookup: example_id -> type
    golden_by_id = {ex.id: ex for ex in golden}

    # Group rows by (provider, model, mode, rerank)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (row["provider"], row["model"], row["mode"], row["rerank"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Build config summaries, sorted for determinism
    config_summaries = []
    for (provider, model, mode, rerank), rows in sorted(groups.items()):
        n_items = len(rows)
        n_judged = sum(1 for r in rows if r["judge"] is not None and not r["refusal"])
        n_refusals = sum(1 for r in rows if r["refusal"])
        n_judge_failures = sum(1 for r in rows if not r["refusal"] and r["judge"] is None)

        # Refusal correctness: refused unanswerable / total unanswerable
        unanswerable_ids = {
            ex_id for ex_id, ex in golden_by_id.items() if ex.type == "unanswerable"
        }
        answerable_ids = {ex_id for ex_id, ex in golden_by_id.items() if ex.type != "unanswerable"}

        refused_unanswerable = sum(
            1 for r in rows if r["refusal"] and r["example_id"] in unanswerable_ids
        )
        total_unanswerable = sum(1 for r in rows if r["example_id"] in unanswerable_ids)
        refusal_correct_rate = (
            refused_unanswerable / total_unanswerable if total_unanswerable > 0 else None
        )

        # False refusals: refused answerable / total answerable
        refused_answerable = sum(
            1 for r in rows if r["refusal"] and r["example_id"] in answerable_ids
        )
        total_answerable = sum(1 for r in rows if r["example_id"] in answerable_ids)
        false_refusal_rate = refused_answerable / total_answerable if total_answerable > 0 else None

        # Scores: means over judged rows
        scores_dict: dict[str, float | None] = {
            "faithfulness": None,
            "relevance": None,
            "citation_accuracy": None,
        }
        judged_rows = [r for r in rows if r["judge"] is not None]
        if judged_rows:
            for dim in ["faithfulness", "relevance", "citation_accuracy"]:
                values = [r["judge"][dim]["score"] for r in judged_rows]
                scores_dict[dim] = round(statistics.mean(values), 4)

        # Latency: mean, p50, p95 of total
        latencies = [r["latency_s"].get("total", 0.0) for r in rows]
        latency_stats = {}
        if latencies:
            latency_stats = {
                "mean": round(statistics.mean(latencies), 4),
                "p50": round(statistics.median(latencies), 4),
                "p95": round(
                    sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0, 4
                ),
            }

        # Gen tokens and cost
        gen_input_sum = sum(r["gen_usage"]["input_tokens"] for r in rows)
        gen_output_sum = sum(r["gen_usage"]["output_tokens"] for r in rows)
        gen_costs = [r["gen_usage"]["cost_usd"] for r in rows]
        gen_cost_total = (
            sum(c for c in gen_costs if c is not None)
            if all(c is not None for c in gen_costs)
            else None
        )

        # Judge tokens and cost (only over judged rows)
        judge_cost_total = None
        if judged_rows:
            judge_costs = [r["judge"]["usage"]["cost_usd"] for r in judged_rows]
            judge_cost_total = (
                sum(c for c in judge_costs if c is not None)
                if all(c is not None for c in judge_costs)
                else None
            )

        # Get synthesis/judge prompt ids
        synthesis_prompt = rows[0].get("synthesis_prompt") if rows else None
        judge_prompt = None
        judge_provider_name = None
        judge_model_name = None
        if judged_rows:
            judge_prompt = judged_rows[0]["judge"].get("prompt_id")
            judge_provider_name = judged_rows[0]["judge"].get("judge_provider")
            judge_model_name = judged_rows[0]["judge"].get("judge_model")

        config_summaries.append(
            {
                "provider": provider,
                "model": model,
                "mode": mode,
                "rerank": rerank,
                "synthesis_prompt": synthesis_prompt,
                "judge_prompt": judge_prompt,
                "judge_provider": judge_provider_name,
                "judge_model": judge_model_name,
                "n_items": n_items,
                "n_judged": n_judged,
                "n_refusals": n_refusals,
                "n_judge_failures": n_judge_failures,
                "refusal_correct_rate": (
                    round(refusal_correct_rate, 4) if refusal_correct_rate is not None else None
                ),
                "false_refusal_rate": (
                    round(false_refusal_rate, 4) if false_refusal_rate is not None else None
                ),
                "scores": scores_dict,
                "latency_s": latency_stats,
                "gen_tokens": {"input": gen_input_sum, "output": gen_output_sum},
                "gen_cost_usd": round(gen_cost_total, 4) if gen_cost_total is not None else None,
                "judge_cost_usd": (
                    round(judge_cost_total, 4) if judge_cost_total is not None else None
                ),
            }
        )

    # Build final summary; rows carry the dataset file stem they were run against
    versions = sorted({str(r["dataset_version"]) for r in all_rows})
    dataset_version = versions[0] if len(versions) == 1 else ("mixed" if versions else "unknown")
    return {
        "run_id": run_dir.name,
        "dataset_version": dataset_version,
        "n_examples": len(golden),
        "configs": config_summaries,
    }
