"""SQLite FTS5 BM25 sparse retrieval index.

Persists chunks as JSON in a single SQLite database with an FTS5 virtual
table over heading and body text. Query preprocessing neutralizes FTS5
operator injection and handles control IDs (AC-2) via phrase queries.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from agentic_rag.retrieval.base import ChunkRecord, ScoredChunk


class BM25Index:
    """BM25 retrieval index backed by SQLite FTS5."""

    def __init__(self, db_path: Path) -> None:
        """Open an existing BM25 index.

        Args:
            db_path: Path to the SQLite database.

        Raises:
            FileNotFoundError: If the database does not exist.
        """
        if not db_path.exists():
            msg = f"BM25 index not found at {db_path}. Run `agentic-rag index` first."
            raise FileNotFoundError(msg)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))

    @classmethod
    def build(cls, chunks: Sequence[ChunkRecord], db_path: Path) -> BM25Index:
        """Build a new BM25 index from chunks.

        Deletes any existing file, creates parent directories, and inserts
        chunks in the given order (deterministic indexing).

        Args:
            chunks: Sequence of ChunkRecord objects in desired order.
            db_path: Path where the database will be created.

        Returns:
            A new BM25Index instance.

        Raises:
            RuntimeError: If FTS5 is unavailable.
        """
        db_path.unlink(missing_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE chunks (
                    rowid INTEGER PRIMARY KEY,
                    json TEXT NOT NULL
                )
                """
            )

            try:
                conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(heading, body)")
            except sqlite3.OperationalError as e:
                msg = (
                    "FTS5 not available in this SQLite build. "
                    "Please rebuild SQLite with --enable-fts5."
                )
                raise RuntimeError(msg) from e

            for idx, chunk in enumerate(chunks, start=1):
                chunk_json = json.dumps(chunk.to_json())
                conn.execute(
                    "INSERT INTO chunks (rowid, json) VALUES (?, ?)",
                    (idx, chunk_json),
                )
                conn.execute(
                    """
                    INSERT INTO chunks_fts (rowid, heading, body)
                    VALUES (?, ?, ?)
                    """,
                    (idx, chunk.heading, chunk.text),
                )

            conn.commit()
        finally:
            conn.close()

        return cls(db_path)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """Search the index for chunks matching the query.

        Args:
            query: User query string.
            top_k: Maximum number of results to return.

        Returns:
            List of ScoredChunk objects ranked by BM25 score.
        """
        fts5_query = self._preprocess_query(query)
        if not fts5_query:
            return []

        cursor = self._conn.execute(
            """
            SELECT c.rowid, c.json, bm25(chunks_fts) as bm25_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC, c.rowid ASC
            LIMIT ?
            """,
            (fts5_query, top_k),
        )

        results = []
        raw_scores = []
        for _rowid, chunk_json, bm25_score in cursor:
            chunk_data = json.loads(chunk_json)
            chunk = ChunkRecord.from_json(chunk_data)
            raw_score = -bm25_score  # FTS5 bm25() is smaller-is-better; flip to positive
            raw_scores.append(raw_score)
            results.append((chunk, raw_score))

        if not raw_scores:
            return []

        max_raw = max(raw_scores)
        scored_chunks = []
        for rank, (chunk, raw_score) in enumerate(results, start=1):
            normalized_score = raw_score / max_raw if max_raw > 0 else 0.0

            scored_chunk = ScoredChunk(
                chunk=chunk,
                score=normalized_score,
                rank=rank,
                source_scores={"bm25": raw_score},
            )
            scored_chunks.append(scored_chunk)

        return scored_chunks

    @property
    def size(self) -> int:
        """Return the number of chunks in the index."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()
        return count[0] if count else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for FTS5 matching.

        Algorithm:
        1. Lowercase the query.
        2. Extract token groups with regex [a-z0-9]+(?:-[a-z0-9]+)*.
        3. Each group becomes a double-quoted FTS5 string.
        4. Groups containing hyphens have them replaced by spaces inside
           the quotes, making them phrase queries.
        5. Join groups with ` OR `.
        6. Empty group list returns empty string.

        Args:
            query: User query string.

        Returns:
            FTS5 query string or empty string if no tokens.
        """
        query_lower = query.lower()
        token_groups = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query_lower)

        if not token_groups:
            return ""

        fts5_terms = [f'"{group.replace("-", " ")}"' for group in token_groups]
        return " OR ".join(fts5_terms)
