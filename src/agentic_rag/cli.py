"""Typer CLI. Thin by design: commands parse arguments and delegate to library
modules so the API service can reuse the same code paths."""

from __future__ import annotations

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
