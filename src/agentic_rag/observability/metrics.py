"""Per-request metrics store: append-only SQLite ledger (ADR-009).

Design: one row per request (CLI/API) or per eval LLM interaction. The ledger
is append-only, indexed on (ts), (provider), (source). No in-process state is
kept — all aggregation happens at query time via ``agentic-rag stats``. This
ensures crashes don't lose data.

Concurrency: WAL journal mode + 5000ms busy_timeout + fresh connection per
operation make concurrent writes from multiple API workers safe. SQLite's ACID
guarantees hold.

Request metrics mirror a subset of audit_v1 fields: the data needed for stats
(tokens, cost, latency, provider, refusal reason) but not the full audit (no
query/answer text, no scan details, no chunk lists — those stay audit-only).
Eval metrics carry the same structure but source="eval.generation" or
source="eval.judge" and may have zero latency/stages for judge rows.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.config import Settings


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (``p`` in [0, 100]); 0.0 for an empty list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    rank = max(0, int((p / 100.0) * len(sorted_vals) + 0.5) - 1)
    return sorted_vals[rank]


@dataclass(frozen=True, slots=True)
class RequestMetric:
    """One row in the metrics ledger.

    Mirrors a subset of AuditRecord fields, augmented with source for routing.
    ``stages`` is stored as JSON text; ``refusal_reason`` is None unless
    refusal is True.
    """

    request_id: str
    ts: str  # ISO-8601 UTC
    source: str  # "cli" | "api" | "eval.generation" | "eval.judge"
    provider: str
    model: str
    pipeline: str  # "vanilla" | "agentic"
    mode: str
    rerank: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float  # total wall-clock seconds
    stages: dict[str, float]  # stage -> seconds, stored as JSON
    refusal: bool
    refusal_reason: str | None
    error: bool = False


class MetricsStore:
    """Append-only SQLite ledger for request metrics.

    Freshly created stores get schema_version=1. Columns exactly match
    RequestMetric fields (stages as TEXT JSON, booleans as INTEGER).
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize store, creating tables and indexes if missing.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with closing(self._connect()) as conn, conn:
            # WAL is persistent (stored in the db file); set once here
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA user_version=1")

            # Create table if missing
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    request_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    pipeline TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    rerank TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL,
                    latency_s REAL NOT NULL,
                    stages TEXT NOT NULL,
                    refusal INTEGER NOT NULL,
                    refusal_reason TEXT,
                    error INTEGER NOT NULL,
                    PRIMARY KEY (request_id, source)
                )
                """
            )

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_provider ON metrics(provider)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_source ON metrics(source)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def record(self, metric: RequestMetric) -> None:
        """Append or replace one metrics row by (request_id, source) key.

        Args:
            metric: The metric row to insert.
        """
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metrics (
                    request_id, source, ts, provider, model, pipeline, mode,
                    rerank, input_tokens, output_tokens, cost_usd, latency_s,
                    stages, refusal, refusal_reason, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.request_id,
                    metric.source,
                    metric.ts,
                    metric.provider,
                    metric.model,
                    metric.pipeline,
                    metric.mode,
                    metric.rerank,
                    metric.input_tokens,
                    metric.output_tokens,
                    metric.cost_usd,
                    metric.latency_s,
                    json.dumps(metric.stages),
                    1 if metric.refusal else 0,
                    metric.refusal_reason,
                    1 if metric.error else 0,
                ),
            )

    def summary(
        self, *, since: str | None = None, group: str = "provider"
    ) -> list[dict[str, object]]:
        """Aggregate metrics by group key.

        Args:
            since: Optional ISO-8601 date string; filters ts >= since.
            group: One of "provider", "model", "day", "source", "pipeline".

        Returns:
            List of dicts, each with: group (key), requests (n), errors (n),
            refusals (n), input_tokens (sum), output_tokens (sum),
            cost_usd (sum or None if all NULL), latency_p50, latency_p95.
        """
        if group not in ("provider", "model", "day", "source", "pipeline"):
            raise ValueError(
                f"Invalid group {group!r}; must be one of provider, model, day, source, pipeline"
            )

        # Map group to column or expression
        group_expr = {
            "provider": "provider",
            "model": "model",
            "day": "substr(ts, 1, 10)",
            "source": "source",
            "pipeline": "pipeline",
        }[group]

        where_clause = " WHERE ts >= ?" if since else ""
        params = (since,) if since else ()

        with closing(self._connect()) as conn:
            # Fetch aggregates
            rows = conn.execute(
                f"""
                SELECT
                    {group_expr} AS group_key,
                    COUNT(*) AS requests,
                    SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) AS errors,
                    SUM(CASE WHEN refusal = 1 THEN 1 ELSE 0 END) AS refusals,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(CASE WHEN cost_usd IS NOT NULL THEN cost_usd ELSE 0 END) AS cost_sum,
                    COUNT(CASE WHEN cost_usd IS NOT NULL THEN 1 END) AS cost_count
                FROM metrics
                {where_clause}
                GROUP BY {group_expr}
                ORDER BY group_key
                """,
                params,
            ).fetchall()

            # Fetch latencies for percentile computation
            latency_rows = conn.execute(
                f"""
                SELECT {group_expr} AS group_key, latency_s
                FROM metrics
                {where_clause}
                ORDER BY {group_expr}, latency_s
                """,
                params,
            ).fetchall()

        # Build latency dict: group_key -> list of latencies
        latencies: dict[str, list[float]] = {}
        for group_key, latency_s in latency_rows:
            if group_key not in latencies:
                latencies[group_key] = []
            latencies[group_key].append(latency_s)

        # Build result rows
        result: list[dict[str, object]] = []
        for (
            group_key,
            requests,
            errors,
            refusals,
            in_tokens,
            out_tokens,
            cost_sum,
            cost_count,
        ) in rows:
            # Determine cost_usd value (sum of non-null or None)
            cost_usd: float | None = None
            if cost_count > 0:
                cost_usd = cost_sum

            # Get latencies for this group
            lats = latencies.get(group_key, [])
            p50 = _percentile(lats, 50)
            p95 = _percentile(lats, 95)

            result.append(
                {
                    "group": group_key,
                    "requests": requests,
                    "errors": errors,
                    "refusals": refusals,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "cost_usd": cost_usd,
                    "latency_p50": p50,
                    "latency_p95": p95,
                }
            )

        return result

    def stage_summary(self, *, since: str | None = None) -> list[dict[str, object]]:
        """Per-stage latency summary across all matching rows.

        Args:
            since: Optional ISO-8601 date string; filters ts >= since.

        Returns:
            List of dicts, each with: stage (name), count (n), p50, p95.
            Sorted by stage name.
        """
        where_clause = " WHERE ts >= ?" if since else ""
        params = (since,) if since else ()

        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT stages FROM metrics{where_clause} ORDER BY ts",
                params,
            ).fetchall()

        # Parse stages JSON and aggregate
        stage_latencies: dict[str, list[float]] = {}
        for (stages_json,) in rows:
            try:
                stages = json.loads(stages_json)
                for stage, seconds in stages.items():
                    if stage not in stage_latencies:
                        stage_latencies[stage] = []
                    stage_latencies[stage].append(seconds)
            except (json.JSONDecodeError, TypeError):
                # Skip malformed entries
                pass

        result: list[dict[str, object]] = []
        for stage in sorted(stage_latencies.keys()):
            lats = stage_latencies[stage]
            result.append(
                {
                    "stage": stage,
                    "count": len(lats),
                    "p50": _percentile(lats, 50),
                    "p95": _percentile(lats, 95),
                }
            )

        return result

    def format_summary(self, rows: list[dict[str, object]], *, group: str) -> str:
        """Format summary rows as a markdown table.

        Args:
            rows: List of dicts from summary().
            group: The grouping key used (for column header).

        Returns:
            Markdown table string.
        """
        if not rows:
            return f"No metrics found for group={group}."

        # Build header and separator
        header = (
            "| " + group.capitalize() + " | Requests | Errors | Refusals | "
            "Input Tokens | Output Tokens | Cost | p50 (s) | p95 (s) |"
        )
        sep = "|" + "|".join([" --- " for _ in range(8)] + [" --- |"])

        # Build data rows
        data_lines: list[str] = []
        for row in rows:
            group_val = row["group"]
            requests = row["requests"]
            errors = row["errors"]
            refusals = row["refusals"]
            in_tokens = row["input_tokens"]
            out_tokens = row["output_tokens"]
            cost_usd = row["cost_usd"]
            p50 = row["latency_p50"]
            p95 = row["latency_p95"]

            cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "n/a"
            in_str = f"{in_tokens:,}" if in_tokens else "0"
            out_str = f"{out_tokens:,}" if out_tokens else "0"

            data_lines.append(
                f"| {group_val} | {requests} | {errors} | {refusals} | {in_str} | "
                f"{out_str} | {cost_str} | {p50:.3f} | {p95:.3f} |"
            )

        return header + "\n" + sep + "\n" + "\n".join(data_lines)

    def format_stages(self, rows: list[dict[str, object]]) -> str:
        """Format stage summary rows as a markdown table.

        Args:
            rows: List of dicts from stage_summary().

        Returns:
            Markdown table string.
        """
        if not rows:
            return "No stage data found."

        header = "| Stage | Count | p50 (s) | p95 (s) |"
        sep = "|" + "|".join([" --- " for _ in range(3)] + [" --- |"])

        data_lines: list[str] = []
        for row in rows:
            stage = row["stage"]
            count = row["count"]
            p50 = row["p50"]
            p95 = row["p95"]
            data_lines.append(f"| {stage} | {count} | {p50:.3f} | {p95:.3f} |")

        return header + "\n" + sep + "\n" + "\n".join(data_lines)


def metrics_store_for(settings: Settings) -> MetricsStore | None:
    """Factory: return a MetricsStore or None if disabled.

    Args:
        settings: Application settings.

    Returns:
        MetricsStore with default or configured db_path, or None if disabled.
    """
    if not settings.metrics.enabled:
        return None

    db_path = settings.metrics.db_path or (settings.data_dir / "metrics.db")
    return MetricsStore(db_path)
