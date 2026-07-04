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
