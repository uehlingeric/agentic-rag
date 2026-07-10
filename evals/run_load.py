#!/usr/bin/env python
"""Concurrency sanity check: N concurrent POST /ask against a running API.

The week-7 exit bar, not a benchmark: 20 concurrent requests on the local
path must complete without corruption, with a coherent p50/p95 latency
report. Corruption checks are structural — every response parses, request_ids
are unique, and non-refusals carry citations. (Ledger integrity — one metrics
row per request — is checked by hand via `agentic-rag stats`; see the review
doc.)

Requires a running API (e.g. `docker compose up` or `agentic-rag serve`) and
its bearer token. Local path is free; pointing this at a paid provider incurs
normal generation cost per request.

Writes evals/results/load-<run-id>/summary.json and prints a markdown report.

Usage:
    AGENTIC_RAG_API__TOKEN=... uv run python evals/run_load.py \\
      [--url http://localhost:8000] [--n 20] [--concurrency 20] \\
      [--question "What does control AC-2 require?"] [--run-id auto]
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import typer

app = typer.Typer()

# Rotated across requests so the run exercises retrieval variety, not one
# cache-friendly query.
_QUESTIONS = [
    "What does control AC-2 require?",
    "What is FIPS 199?",
    "How does SP 800-171 relate to SP 800-53?",
    "What are the security objectives defined in FIPS 199?",
    "What does the AI Risk Management Framework say about governance?",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, int((p / 100.0) * len(ordered) + 0.5) - 1)
    return ordered[rank]


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    question: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, object]:
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post(
                f"{url}/ask",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": question},
            )
            elapsed = time.perf_counter() - start
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc), "elapsed_s": time.perf_counter() - start}

    if response.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
            "elapsed_s": elapsed,
        }

    body = response.json()
    return {
        "ok": True,
        "elapsed_s": elapsed,
        "request_id": body.get("request_id"),
        "refusal": body.get("refusal"),
        "n_citations": len(body.get("citations", [])),
        "answer_sha_prefix": hash(body.get("answer", "")) & 0xFFFF,
    }


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000", help="Base URL of a running API."),
    token: str | None = typer.Option(None, help="Bearer token (or AGENTIC_RAG_API__TOKEN env)."),
    n: int = typer.Option(20, help="Total requests."),
    concurrency: int = typer.Option(20, help="Concurrent in-flight requests."),
    question: str | None = typer.Option(None, help="Single question (default: rotate 5)."),
    run_id: str | None = typer.Option(None, help="Results dir suffix (default: timestamp)."),
) -> None:
    """Fire N concurrent /ask requests and report latency + integrity."""
    resolved_token = token or os.environ.get("AGENTIC_RAG_API__TOKEN")
    if not resolved_token:
        typer.secho("no token: pass --token or set AGENTIC_RAG_API__TOKEN", fg="red", err=True)
        raise typer.Exit(1)

    questions = [question] * n if question else [_QUESTIONS[i % len(_QUESTIONS)] for i in range(n)]

    async def _run() -> tuple[list[dict[str, object]], float]:
        semaphore = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=600.0) as client:
            results = await asyncio.gather(
                *(_one_request(client, url, resolved_token, q, semaphore) for q in questions)
            )
        return list(results), time.perf_counter() - started

    results, wall_s = asyncio.run(_run())

    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    latencies = [float(r["elapsed_s"]) for r in ok]
    request_ids = [r["request_id"] for r in ok]
    non_refusals = [r for r in ok if not r["refusal"]]

    integrity = {
        "unique_request_ids": len(set(request_ids)) == len(request_ids),
        "non_refusals_have_citations": all(int(r["n_citations"]) > 0 for r in non_refusals),  # type: ignore[call-overload]
    }

    summary = {
        "run_id": run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ"),
        "url": url,
        "n": n,
        "concurrency": concurrency,
        "wall_s": round(wall_s, 3),
        "succeeded": len(ok),
        "failed": len(errors),
        "refusals": sum(1 for r in ok if r["refusal"]),
        "latency_s": {
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
        },
        "integrity": integrity,
        "errors": [str(r["error"]) for r in errors],
    }

    out_dir = Path("evals/results") / f"load-{summary['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    lat = summary["latency_s"]
    typer.echo("| Requests | Concurrency | OK | Failed | Refusals | p50 (s) | p95 (s) | Wall (s) |")
    typer.echo("| --- | --- | --- | --- | --- | --- | --- | --- |")
    typer.echo(
        f"| {n} | {concurrency} | {len(ok)} | {len(errors)} | {summary['refusals']} "
        f"| {lat['p50']} | {lat['p95']} | {summary['wall_s']} |"  # type: ignore[index]
    )
    typer.echo(f"\nintegrity: {integrity}")
    typer.echo(f"results written to {out_path}")

    if errors or not all(integrity.values()):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
