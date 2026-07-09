"""Typer CLI. Thin by design: commands parse arguments and delegate to library
modules so the API service can reuse the same code paths. Imports of library
modules stay inside command bodies to keep CLI startup fast."""

from __future__ import annotations

import asyncio

import typer

import agentic_rag

app = typer.Typer(
    name="agentic-rag",
    help="Provider-agnostic agentic RAG reference system.",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(agentic_rag.__version__)
        raise typer.Exit()


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    provider: str | None = typer.Option(None, help="Provider: anthropic|openai|google|ollama."),
    model: str | None = typer.Option(None, help="Model override for the chosen provider."),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they arrive."),
) -> None:
    """Send a single (non-RAG) prompt to a configured provider."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers import Message, Role
    from agentic_rag.providers.registry import get_llm_provider

    settings = get_settings()
    llm = get_llm_provider(provider or settings.provider, settings)
    messages = [Message(role=Role.USER, content=prompt)]

    async def _run() -> None:
        if stream:
            usage_line = ""
            async for event in llm.stream(messages, model=model):
                if event.completion is not None:
                    u = event.completion.usage
                    usage_line = _usage_line(event.completion.model, u)
                else:
                    typer.echo(event.delta, nl=False)
            typer.echo()
            typer.secho(usage_line, fg="bright_black", err=True)
        else:
            completion = await llm.complete(messages, model=model)
            typer.echo(completion.text)
            typer.secho(
                _usage_line(completion.model, completion.usage), fg="bright_black", err=True
            )

    asyncio.run(_run())


def _usage_line(model: str, u: agentic_rag.providers.Usage) -> str:
    cost = f"${u.cost_usd:.4f}" if u.cost_usd is not None else "n/a"
    return f"[{model}] {u.input_tokens} in / {u.output_tokens} out / cost {cost}"


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed corpus."),
    provider: str | None = typer.Option(None, help="Provider: anthropic|openai|google|ollama."),
    mode: str = typer.Option("hybrid", help="Retrieval mode: bm25|dense|hybrid."),
    rerank: str | None = typer.Option(
        None, help="Rerank stage: none|llm|cross-encoder (default: config)."
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they arrive."),
    json_out: bool = typer.Option(False, "--json", help="Emit a JSON record for scripting."),
) -> None:
    """Answer a question from the corpus with inline citations."""
    import json

    from agentic_rag.config import get_settings
    from agentic_rag.pipeline.base import Answer
    from agentic_rag.pipeline.pipeline import RAGPipeline
    from agentic_rag.providers.registry import get_embedding_provider, get_llm_provider
    from agentic_rag.rerank.base import NoopReranker, Reranker
    from agentic_rag.retrieval.base import RetrievalMode
    from agentic_rag.retrieval.retriever import Retriever

    if stream and json_out:
        raise typer.BadParameter("--stream and --json are mutually exclusive.")

    settings = get_settings()
    llm = get_llm_provider(provider or settings.provider, settings)

    rerank_mode = rerank if rerank is not None else settings.rerank.mode
    reranker: Reranker
    if rerank_mode == "none":
        reranker = NoopReranker()
    elif rerank_mode == "llm":
        from agentic_rag.rerank.llm import LLMReranker

        reranker = LLMReranker(llm, model=settings.rerank.model)
    elif rerank_mode == "cross-encoder":
        from agentic_rag.rerank.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model=settings.rerank.model)
    else:
        raise typer.BadParameter(
            f"Unknown rerank mode: {rerank_mode}. Valid options: none|llm|cross-encoder."
        )

    retriever = Retriever.load(
        settings.data_dir / "index",
        get_embedding_provider(settings.embedding.provider, settings),
        rrf_k=settings.retrieval.rrf_k,
        candidate_pool=settings.retrieval.candidate_pool,
    )
    pipeline = RAGPipeline(retriever, reranker, llm, settings)
    retrieval_mode = RetrievalMode(mode)

    def _record(answer: Answer) -> dict[str, object]:
        return {
            "question": question,
            "provider": llm.name,
            "mode": retrieval_mode.value,
            "rerank": reranker.name,
            "answer": answer.text,
            "refusal": answer.refusal,
            "citations": [
                {
                    "marker": c.marker,
                    "chunk_id": c.chunk.chunk_id,
                    "doc_id": c.chunk.doc_id,
                    "section_id": c.chunk.section_id,
                    "heading": c.chunk.heading,
                    "page_start": c.chunk.page_start,
                }
                for c in answer.citations
            ],
            "invalid_citations": answer.invalid_citations,
            "usage": {
                "input_tokens": answer.usage.input_tokens,
                "output_tokens": answer.usage.output_tokens,
                "cost_usd": answer.usage.cost_usd,
            },
            "timings": {t.stage: t.seconds for t in answer.timings},
        }

    def _footer(answer: Answer) -> None:
        if answer.citations:
            typer.echo()
            for c in answer.citations:
                typer.echo(
                    f"[{c.marker}] {c.chunk.doc_id} §{c.chunk.section_id} (p.{c.chunk.page_start})"
                )
        if answer.invalid_citations:
            typer.secho(
                f"warning: stripped invalid citation markers {answer.invalid_citations}",
                fg="yellow",
                err=True,
            )
        timings = " | ".join(f"{t.stage} {t.seconds:.2f}s" for t in answer.timings)
        typer.secho(f"[{timings}]", fg="bright_black", err=True)
        typer.secho(_usage_line(llm.name, answer.usage), fg="bright_black", err=True)

    async def _run() -> None:
        if stream:
            final: Answer | None = None
            async for event in pipeline.ask_stream(question, mode=retrieval_mode):
                if event.answer is not None:
                    final = event.answer
                else:
                    typer.echo(event.delta, nl=False)
            typer.echo()
            if final is None:
                return
            if final.refusal:
                typer.secho("Not found in corpus.", fg="yellow")
            _footer(final)
        else:
            answer = await pipeline.ask(question, mode=retrieval_mode)
            if json_out:
                typer.echo(json.dumps(_record(answer), indent=2))
                return
            if answer.refusal:
                msg = "Not found in corpus." + (f" {answer.text}" if answer.text else "")
                typer.secho(msg, fg="yellow")
            else:
                typer.echo(answer.text)
            _footer(answer)

    asyncio.run(_run())


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Re-embed and rebuild everything."),
) -> None:
    """Build the BM25 and dense indexes from the chunked corpus."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_embedding_provider
    from agentic_rag.retrieval.base import load_chunks
    from agentic_rag.retrieval.bm25 import BM25Index
    from agentic_rag.retrieval.dense import DenseIndex
    from agentic_rag.retrieval.embed import embed_corpus

    settings = get_settings()
    index_dir = settings.data_dir / "index"
    chunks = load_chunks(settings.data_dir / "corpus" / "chunks.jsonl")

    bm25 = BM25Index.build(chunks, index_dir / "bm25.db")
    typer.echo(f"BM25 index: {bm25.size} chunks -> {index_dir / 'bm25.db'}")
    bm25.close()

    embedder = get_embedding_provider(settings.embedding.provider, settings)

    async def _embed() -> None:
        matrix = await embed_corpus(
            chunks,
            embedder,
            model=settings.embedding.model,
            checkpoint_path=index_dir / "embeddings.checkpoint.jsonl",
            force=force,
        )
        dense = DenseIndex.build(matrix, chunks, index_dir)
        typer.echo(
            f"Dense index: {dense.size} chunks, {dense.manifest.dimensions} dims "
            f"({dense.manifest.model}) -> {index_dir}"
        )

    asyncio.run(_embed())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    mode: str = typer.Option("hybrid", help="Retrieval mode: bm25|dense|hybrid."),
    top_k: int = typer.Option(None, "--top-k", help="Number of results."),
) -> None:
    """Search the indexed corpus and print ranked chunks."""
    from agentic_rag.config import get_settings
    from agentic_rag.providers.registry import get_embedding_provider
    from agentic_rag.retrieval.base import RetrievalMode
    from agentic_rag.retrieval.retriever import Retriever

    settings = get_settings()
    retriever = Retriever.load(
        settings.data_dir / "index",
        get_embedding_provider(settings.embedding.provider, settings),
        rrf_k=settings.retrieval.rrf_k,
        candidate_pool=settings.retrieval.candidate_pool,
    )
    k = top_k if top_k is not None else settings.retrieval.top_k

    async def _run() -> None:
        results = await retriever.retrieve(query, mode=RetrievalMode(mode), top_k=k)
        for r in results:
            typer.echo(
                f"{r.rank:>2}. [{r.score:.4f}] {r.chunk.doc_id}:{r.chunk.section_id} "
                f"(p.{r.chunk.page_start}) {r.chunk.heading[:60]}"
            )
            typer.secho(f"    {r.chunk.text[:160]}", fg="bright_black")

    asyncio.run(_run())


eval_app = typer.Typer(help="Evaluation harnesses.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: str = typer.Option("evals/golden/v1.jsonl", help="Golden dataset JSONL path."),
) -> None:
    """Run the retrieval-only eval over the golden dataset."""
    import json
    from pathlib import Path

    from agentic_rag.config import get_settings
    from agentic_rag.evals.retrieval import (
        load_golden,
        report_markdown,
        run_eval,
        write_results,
    )
    from agentic_rag.providers.registry import get_embedding_provider
    from agentic_rag.retrieval.retriever import Retriever

    settings = get_settings()
    index_dir = settings.data_dir / "index"
    retriever = Retriever.load(
        index_dir,
        get_embedding_provider(settings.embedding.provider, settings),
        rrf_k=settings.retrieval.rrf_k,
        candidate_pool=settings.retrieval.candidate_pool,
    )
    golden = load_golden(Path(dataset))
    manifest = json.loads((index_dir / "manifest.json").read_text())

    async def _run() -> None:
        report = await run_eval(
            retriever,
            golden,
            modes=["bm25", "dense", "hybrid"],
            config={
                "dataset": dataset,
                "n_questions": len(golden),
                "corpus_fingerprint": manifest["fingerprint"],
                "embedding_model": manifest["model"],
                "rrf_k": settings.retrieval.rrf_k,
                "candidate_pool": settings.retrieval.candidate_pool,
                "top_k": 20,
            },
        )
        typer.echo(report_markdown(report))
        path = write_results(report, Path("evals/results"))
        typer.echo(f"Results written to {path}")

    asyncio.run(_run())


@eval_app.command("rerank")
def eval_rerank(
    dataset: str = typer.Option("evals/golden/v1.jsonl", help="Golden dataset JSONL path."),
    provider: str | None = typer.Option(None, help="LLM provider for the llm reranker."),
    reranker: str = typer.Option("llm", help="Reranker to evaluate: llm|cross-encoder."),
    pool: int = typer.Option(30, help="Candidate pool retrieved per query."),
    top_k: int = typer.Option(10, help="Ranking depth for metrics and rerank cut."),
) -> None:
    """Run the rerank on/off eval over the golden dataset."""
    import json
    from pathlib import Path

    from agentic_rag.config import get_settings
    from agentic_rag.evals.rerank import run_rerank_eval
    from agentic_rag.evals.retrieval import load_golden, report_markdown, write_results
    from agentic_rag.prompts import load_prompt
    from agentic_rag.providers.registry import get_embedding_provider, get_llm_provider
    from agentic_rag.rerank.base import Reranker
    from agentic_rag.retrieval.retriever import Retriever

    settings = get_settings()
    provider_name = provider or settings.provider

    rr: Reranker
    config: dict[str, object] = {}
    if reranker == "llm":
        from agentic_rag.rerank.llm import LLMReranker

        rr = LLMReranker(get_llm_provider(provider_name, settings), model=settings.rerank.model)
        config["reranker_provider"] = provider_name
        config["rerank_prompt"] = load_prompt("rerank").id
    elif reranker == "cross-encoder":
        from agentic_rag.rerank.cross_encoder import CrossEncoderReranker

        rr = CrossEncoderReranker(model=settings.rerank.model)
    else:
        raise typer.BadParameter(f"Unknown reranker: {reranker}. Valid: llm|cross-encoder.")

    index_dir = settings.data_dir / "index"
    retriever = Retriever.load(
        index_dir,
        get_embedding_provider(settings.embedding.provider, settings),
        rrf_k=settings.retrieval.rrf_k,
        candidate_pool=settings.retrieval.candidate_pool,
    )
    golden = load_golden(Path(dataset))
    manifest = json.loads((index_dir / "manifest.json").read_text())
    config.update(
        {
            "dataset": dataset,
            "n_questions": len(golden),
            "corpus_fingerprint": manifest["fingerprint"],
            "embedding_model": manifest["model"],
            "rrf_k": settings.retrieval.rrf_k,
        }
    )

    async def _run() -> None:
        report = await run_rerank_eval(
            retriever,
            rr,
            golden,
            modes=["bm25", "dense", "hybrid"],
            pool=pool,
            top_k=top_k,
            config=config,
        )
        typer.echo(report_markdown(report))
        path = write_results(report, Path("evals/results"), prefix="rerank")
        typer.echo(f"Results written to {path}")

    asyncio.run(_run())


@app.command()
def ingest(
    doc: list[str] = typer.Option(  # noqa: B008
        None, "--doc", help="Restrict to specific doc ids (default: full corpus)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-download and re-chunk everything."),
) -> None:
    """Download the NIST corpus, extract text, and write chunked JSONL."""
    from agentic_rag.config import get_settings
    from agentic_rag.ingest.pipeline import run_ingest

    settings = get_settings()
    manifest = run_ingest(settings, doc_ids=doc or None, force=force)
    typer.echo(
        f"Ingested {len(manifest.documents)} documents, "
        f"{manifest.total_chunks} chunks -> {manifest.output_path}"
    )
