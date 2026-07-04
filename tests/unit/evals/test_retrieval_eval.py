"""Tests for retrieval evaluation metrics and aggregation."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from agentic_rag.evals.retrieval import (
    Citation,
    EvalReport,
    GoldenExample,
    evaluate_ranking,
    load_golden,
    report_json,
    report_markdown,
    run_eval,
    write_results,
)
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


def make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc1",
    section_id: str = "S1",
    section_ids: list[str] | None = None,
    text: str = "chunk text",
) -> ChunkRecord:
    """Helper to create a ChunkRecord for testing."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        section_ids=section_ids if section_ids is not None else [section_id],
        section_path="section",
        heading="heading",
        page_start=1,
        page_end=2,
        token_count=10,
        text=text,
    )


def make_scored(chunk: ChunkRecord, rank: int, score: float = 1.0) -> ScoredChunk:
    """Helper to create a ScoredChunk."""
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


class TestEvaluateRankingFixtures:
    """Hand-verified fixture tests with explicit metric calculation."""

    def test_single_citation_rank_1(self) -> None:
        """Single citation, covering chunk at rank 1.

        Metrics:
        - recall@5 = 1/1 = 1.0
        - recall@10 = 1/1 = 1.0
        - recall@20 = 1/1 = 1.0
        - precision@5 = 1/5 = 0.2
        - mrr = 1/1 = 1.0
        - ndcg@10: DCG = 1/log2(2) = 1/1 = 1.0; IDCG = 1/log2(2) = 1.0; ndcg = 1.0
        """
        citations = [Citation("doc1", "S1")]
        ranking = [
            make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1),
        ]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)
        assert metrics["recall@20"] == pytest.approx(1.0)
        assert metrics["precision@5"] == pytest.approx(0.2)
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["ndcg@10"] == pytest.approx(1.0)

    def test_single_citation_rank_3(self) -> None:
        """Single citation, covering chunk at rank 3.

        Metrics:
        - recall@5 = 1/1 = 1.0 (covered in top-5)
        - recall@10 = 1/1 = 1.0
        - recall@20 = 1/1 = 1.0
        - precision@5 = 1/5 = 0.2
        - mrr = 1/3 ≈ 0.333...
        - ndcg@10: DCG = 1/log2(4) = 1/2 = 0.5; IDCG = 1/log2(2) = 1.0; ndcg = 0.5
        """
        citations = [Citation("doc1", "S1")]
        ranking = [
            make_scored(make_chunk("c1", doc_id="doc2"), rank=1),
            make_scored(make_chunk("c2", doc_id="doc2"), rank=2),
            make_scored(make_chunk("c3", doc_id="doc1", section_ids=["S1"]), rank=3),
        ]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)
        assert metrics["recall@20"] == pytest.approx(1.0)
        assert metrics["precision@5"] == pytest.approx(0.2)
        assert metrics["mrr"] == pytest.approx(1.0 / 3.0)
        assert metrics["ndcg@10"] == pytest.approx(0.5)

    def test_two_citations_one_covered(self) -> None:
        """Two citations, one covered at rank 1, other never.

        Metrics:
        - recall@5 = 1/2 = 0.5
        - recall@10 = 1/2 = 0.5
        - recall@20 = 1/2 = 0.5
        - precision@5 = 1/5 = 0.2 (only 1 chunk in top-5 covers >= 1 citation)
        - mrr = 1/1 = 1.0
        - ndcg@10: DCG = 1/log2(2) = 1.0
          IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/1.585 ≈ 1.0 + 0.631 = 1.631
          ndcg ≈ 1.0 / 1.631 ≈ 0.613
        """
        citations = [Citation("doc1", "S1"), Citation("doc2", "S2")]
        ranking = [
            make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1),
        ] + [make_scored(make_chunk(f"c{i}", doc_id="doc3"), rank=i) for i in range(2, 21)]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(0.5)
        assert metrics["recall@10"] == pytest.approx(0.5)
        assert metrics["recall@20"] == pytest.approx(0.5)
        assert metrics["precision@5"] == pytest.approx(0.2)
        assert metrics["mrr"] == pytest.approx(1.0)

        # NDCG: DCG = 1/log2(2) = 1; IDCG = 1/log2(2) + 1/log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected_ndcg = 1.0 / idcg
        assert metrics["ndcg@10"] == pytest.approx(expected_ndcg)

    def test_no_covering_chunks(self) -> None:
        """No covering chunks.

        All metrics should be 0.
        """
        citations = [Citation("doc1", "S1")]
        ranking = [make_scored(make_chunk(f"c{i}", doc_id="doc2"), rank=i) for i in range(1, 6)]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(0.0)
        assert metrics["recall@10"] == pytest.approx(0.0)
        assert metrics["recall@20"] == pytest.approx(0.0)
        assert metrics["precision@5"] == pytest.approx(0.0)
        assert metrics["mrr"] == pytest.approx(0.0)
        assert metrics["ndcg@10"] == pytest.approx(0.0)

    def test_two_citations_same_chunk_rank_2(self) -> None:
        """Two citations covered by same chunk at rank 2.

        Citations: [("doc1", "S1"), ("doc1", "S2")]
        Chunk at rank 2 has section_ids = ["S1", "S2"]

        Metrics:
        - recall@5 = 2/2 = 1.0 (both covered in top-5)
        - recall@10 = 2/2 = 1.0
        - recall@20 = 2/2 = 1.0
        - precision@5 = 1/5 = 0.2 (only 1 chunk covers >= 1 citation)
        - mrr = 1/2 = 0.5
        - ndcg@10: DCG = 1/log2(3) ≈ 0.631
          IDCG = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.631 = 1.631
          ndcg ≈ 0.631 / 1.631 ≈ 0.387
        """
        citations = [Citation("doc1", "S1"), Citation("doc1", "S2")]
        ranking = [
            make_scored(make_chunk("c1", doc_id="doc2"), rank=1),
            make_scored(make_chunk("c2", doc_id="doc1", section_ids=["S1", "S2"]), rank=2),
        ]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)
        assert metrics["recall@20"] == pytest.approx(1.0)
        assert metrics["precision@5"] == pytest.approx(0.2)
        assert metrics["mrr"] == pytest.approx(0.5)

        # NDCG: DCG = 1/log2(3); IDCG = 1/log2(2) + 1/log2(3)
        dcg = 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected_ndcg = dcg / idcg
        assert metrics["ndcg@10"] == pytest.approx(expected_ndcg)

    def test_duplicate_coverage_does_not_inflate_ndcg(self) -> None:
        """One citation covered by chunks at BOTH rank 1 and rank 2.

        Repeat coverage of an already-covered citation earns no gain
        (novelty-binary convention), so NDCG stays capped at 1.0. Without
        the novelty rule this would compute DCG = 1/log2(2) + 1/log2(3)
        = 1.631 against IDCG = 1.0, i.e. an impossible NDCG of 1.631.

        Metrics:
        - recall@k = 1/1 = 1.0
        - precision@5 = 2/5 = 0.4 (both chunks cover the citation)
        - mrr = 1/1 = 1.0
        - ndcg@10: DCG = 1/log2(2) = 1.0 (rank 2 adds nothing new);
          IDCG = 1/log2(2) = 1.0; ndcg = 1.0
        """
        citations = [Citation("doc1", "S1")]
        ranking = [
            make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1),
            make_scored(make_chunk("c2", doc_id="doc1", section_ids=["S1"]), rank=2),
        ]

        metrics = evaluate_ranking(citations, ranking)

        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["precision@5"] == pytest.approx(0.4)
        assert metrics["mrr"] == pytest.approx(1.0)
        assert metrics["ndcg@10"] == pytest.approx(1.0)


class TestLoadGolden:
    """Tests for load_golden parsing and validation."""

    def test_load_golden_roundtrip(self) -> None:
        """Load golden from JSONL, verify schema, roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_file = Path(tmpdir) / "golden.jsonl"
            examples = [
                {
                    "id": "q1",
                    "question": "What is A?",
                    "reference_answer": "A is ...",
                    "source_citations": [
                        {"doc": "doc1", "section": "S1"},
                        {"doc": "doc2", "section": "S2"},
                    ],
                    "difficulty": "easy",
                    "type": "lookup",
                },
                {
                    "id": "q2",
                    "question": "Unanswerable?",
                    "reference_answer": "Cannot answer.",
                    "source_citations": [],
                    "difficulty": "hard",
                    "type": "unanswerable",
                },
            ]
            with golden_file.open("w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            loaded = load_golden(golden_file)

            assert len(loaded) == 2
            assert loaded[0].id == "q1"
            assert loaded[0].question == "What is A?"
            assert len(loaded[0].source_citations) == 2
            assert loaded[0].source_citations[0].doc == "doc1"
            assert loaded[0].source_citations[0].section == "S1"
            assert loaded[0].type == "lookup"

            assert loaded[1].id == "q2"
            assert loaded[1].type == "unanswerable"
            assert len(loaded[1].source_citations) == 0

    def test_load_golden_file_not_found(self) -> None:
        """FileNotFoundError if file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_golden(Path("/nonexistent/path.jsonl"))

    def test_load_golden_invalid_schema(self) -> None:
        """ValueError on missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_file = Path(tmpdir) / "golden.jsonl"
            bad_row = {"id": "q1", "question": "What?"}  # missing required fields
            with golden_file.open("w") as f:
                f.write(json.dumps(bad_row) + "\n")

            with pytest.raises(ValueError, match="missing keys"):
                load_golden(golden_file)


class TestReportFormatting:
    """Tests for report formatting."""

    def test_report_markdown(self) -> None:
        """report_markdown produces table with 4-decimal values."""
        from agentic_rag.evals.retrieval import ModeReport

        mode_report = ModeReport(
            mode="bm25",
            metrics={
                "recall@5": 0.1234,
                "recall@10": 0.2345,
                "recall@20": 0.3456,
                "precision@5": 0.4567,
                "mrr": 0.5678,
                "ndcg@10": 0.6789,
            },
            per_query={},
        )
        report = EvalReport(
            modes=[mode_report],
            n_answerable=10,
            n_skipped_unanswerable=5,
            config={"dataset": "golden.jsonl", "n_questions": 15},
        )

        md = report_markdown(report)

        assert "| Mode | Recall@5" in md
        assert "| bm25 |" in md
        assert "0.1234" in md
        assert "0.2345" in md
        assert "0.3456" in md
        assert "0.4567" in md
        assert "0.5678" in md
        assert "0.6789" in md
        assert "10" in md
        assert "5" in md

    def test_report_json_roundtrip(self) -> None:
        """report_json produces JSON-serializable dict."""
        from agentic_rag.evals.retrieval import ModeReport

        mode_report = ModeReport(
            mode="hybrid",
            metrics={
                "recall@5": 0.5,
                "recall@10": 0.6,
                "recall@20": 0.7,
                "precision@5": 0.2,
                "mrr": 0.8,
                "ndcg@10": 0.9,
            },
            per_query={
                "q1": {
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "recall@20": 1.0,
                    "precision@5": 0.2,
                    "mrr": 1.0,
                    "ndcg@10": 1.0,
                }
            },
        )
        report = EvalReport(
            modes=[mode_report],
            n_answerable=1,
            n_skipped_unanswerable=0,
            config={},
        )

        data = report_json(report)

        # Should be JSON-serializable
        json_str = json.dumps(data)
        reloaded = json.loads(json_str)

        assert reloaded["n_answerable"] == 1
        assert reloaded["n_skipped_unanswerable"] == 0
        assert len(reloaded["modes"]) == 1
        assert reloaded["modes"][0]["mode"] == "hybrid"
        assert reloaded["modes"][0]["per_query"]["q1"]["recall@5"] == 1.0


class TestWriteResults:
    """Tests for write_results file I/O."""

    def test_write_results_creates_file(self) -> None:
        """write_results creates a timestamped JSON file."""
        from agentic_rag.evals.retrieval import ModeReport

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            mode_report = ModeReport(
                mode="bm25",
                metrics={
                    "recall@5": 0.5,
                    "recall@10": 0.6,
                    "recall@20": 0.7,
                    "precision@5": 0.2,
                    "mrr": 0.8,
                    "ndcg@10": 0.9,
                },
                per_query={},
            )
            report = EvalReport(
                modes=[mode_report],
                n_answerable=1,
                n_skipped_unanswerable=0,
                config={},
            )

            filepath = write_results(report, results_dir)

            assert filepath.exists()
            assert "retrieval-" in filepath.name
            assert filepath.name.endswith(".json")
            assert filepath.parent == results_dir

            # Verify content
            with filepath.open() as f:
                data = json.load(f)
            assert data["n_answerable"] == 1
            assert data["modes"][0]["mode"] == "bm25"


class TestRunEval:
    """Tests for run_eval aggregation and filtering."""

    class StubRetriever:
        """Stub retriever for testing (no async calls to real index)."""

        def __init__(self, rankings: dict[tuple[str, str], list[ScoredChunk]]) -> None:
            """Init with pre-canned rankings.

            Args:
                rankings: Maps (query_text, mode_name) -> list[ScoredChunk].
            """
            self.rankings = rankings

        async def retrieve(
            self, query: str, *, mode: RetrievalMode, top_k: int
        ) -> list[ScoredChunk]:
            """Return pre-canned ranking for (query, mode)."""
            key = (query, mode.value)
            return self.rankings.get(key, [])

    @pytest.mark.asyncio
    async def test_run_eval_macro_average(self) -> None:
        """run_eval correctly macro-averages over answerable examples."""
        golden = [
            GoldenExample(
                id="q1",
                question="Question 1?",
                reference_answer="Answer 1",
                source_citations=[Citation("doc1", "S1")],
                difficulty="easy",
                type="lookup",
            ),
            GoldenExample(
                id="q2",
                question="Question 2?",
                reference_answer="Answer 2",
                source_citations=[Citation("doc2", "S2")],
                difficulty="easy",
                type="lookup",
            ),
            GoldenExample(
                id="q3",
                question="Unanswerable question?",
                reference_answer="Cannot answer",
                source_citations=[],
                difficulty="hard",
                type="unanswerable",
            ),
        ]

        # Set up stub rankings:
        # q1 (bm25): rank 1 covers [C("doc1", "S1")] ->
        #   recall@5=1, precision@5=0.2, mrr=1, ndcg@10=1
        # q2 (bm25): rank 3 covers [C("doc2", "S2")] ->
        #   recall@5=1, precision@5=0.2, mrr=0.333, ndcg@10=0.5
        # q3 should be skipped (unanswerable)

        rankings = {
            ("Question 1?", "bm25"): [
                make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1),
            ],
            ("Question 2?", "bm25"): [
                make_scored(make_chunk("c1", doc_id="doc3"), rank=1),
                make_scored(make_chunk("c2", doc_id="doc3"), rank=2),
                make_scored(make_chunk("c3", doc_id="doc2", section_ids=["S2"]), rank=3),
            ],
        }

        retriever = self.StubRetriever(rankings)
        report = await run_eval(retriever, golden, modes=["bm25"])

        # Should only evaluate 2 answerable examples
        assert report.n_answerable == 2
        assert report.n_skipped_unanswerable == 1

        # Macro-average: recall@5 = (1 + 1) / 2 = 1.0
        mode_report = report.modes[0]
        assert mode_report.mode == "bm25"
        assert mode_report.metrics["recall@5"] == pytest.approx(1.0)
        assert mode_report.metrics["precision@5"] == pytest.approx(0.2)

        # For mrr: (1 + 1/3) / 2 = 0.667
        expected_mrr = (1.0 + 1.0 / 3.0) / 2.0
        assert mode_report.metrics["mrr"] == pytest.approx(expected_mrr)

        # For ndcg@10: (1.0 + 0.5) / 2 = 0.75
        assert mode_report.metrics["ndcg@10"] == pytest.approx(0.75)

        # per_query should have entries for q1, q2 but not q3
        assert "q1" in mode_report.per_query
        assert "q2" in mode_report.per_query
        assert "q3" not in mode_report.per_query

    @pytest.mark.asyncio
    async def test_run_eval_multiple_modes(self) -> None:
        """run_eval evaluates multiple modes in order."""
        golden = [
            GoldenExample(
                id="q1",
                question="Question 1?",
                reference_answer="Answer 1",
                source_citations=[Citation("doc1", "S1")],
                difficulty="easy",
                type="lookup",
            ),
        ]

        rankings = {
            ("Question 1?", "bm25"): [
                make_scored(make_chunk("c1", doc_id="doc1", section_ids=["S1"]), rank=1),
            ],
            ("Question 1?", "dense"): [
                make_scored(make_chunk("c2", doc_id="doc1", section_ids=["S1"]), rank=1),
            ],
        }

        retriever = self.StubRetriever(rankings)
        report = await run_eval(retriever, golden, modes=["bm25", "dense"])

        assert len(report.modes) == 2
        assert report.modes[0].mode == "bm25"
        assert report.modes[1].mode == "dense"
        # Both should have perfect recall@5
        assert report.modes[0].metrics["recall@5"] == pytest.approx(1.0)
        assert report.modes[1].metrics["recall@5"] == pytest.approx(1.0)
