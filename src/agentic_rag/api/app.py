"""FastAPI application factory and endpoint handlers.

No ``from __future__ import annotations`` here: route annotations reference
closure-locals (the auth dependency), and FastAPI resolves stringified
annotations against module globals only — eager evaluation keeps Depends
working.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import agentic_rag
from agentic_rag.agent.graph import AgenticPipeline
from agentic_rag.api.auth import bearer_auth
from agentic_rag.api.errors import (
    general_exception_handler,
    http_exception_handler,
    rate_limit_handler,
    validation_error_handler,
)
from agentic_rag.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    GuardrailScanOut,
    GuardrailsOut,
    HealthResponse,
    SearchResult,
    StageRow,
    StatsResponse,
    StatsRow,
    UsageOut,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.guardrails.guarded import (
    GuardedPipeline,
    GuardedResult,
    provider_model,
)
from agentic_rag.observability import setup_tracing
from agentic_rag.observability.metrics import MetricsStore
from agentic_rag.pipeline.pipeline import RAGPipeline, SupportsRetrieve
from agentic_rag.providers.base import ProviderError
from agentic_rag.providers.registry import get_embedding_provider, get_llm_provider
from agentic_rag.rerank.base import NoopReranker, Reranker
from agentic_rag.rerank.cross_encoder import CrossEncoderReranker
from agentic_rag.rerank.llm import LLMReranker
from agentic_rag.retrieval.base import RetrievalMode
from agentic_rag.retrieval.retriever import Retriever

logger = logging.getLogger("agentic_rag.api")


def create_app(
    settings: Settings | None = None, *, retriever: SupportsRetrieve | None = None
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings (default: loaded from environment).
        retriever: Retriever to use (default: loaded from data_dir/index).

    Returns:
        Configured FastAPI application.

    Raises:
        RuntimeError: If settings.api.token is unset or index files missing.
    """
    if settings is None:
        settings = get_settings()

    # Fail fast: token is required
    if settings.api.token is None:
        raise RuntimeError("AGENTIC_RAG_API__TOKEN must be set")

    # Shared retriever (loaded once)
    _retriever: SupportsRetrieve | None = retriever

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        """Application lifespan: startup and shutdown."""
        # Startup
        setup_tracing(settings)

        # Load retriever if not injected
        nonlocal _retriever
        if _retriever is None:
            index_dir = settings.data_dir / "index"
            if not (index_dir / "manifest.json").exists():
                raise RuntimeError(
                    f"Index files not found at {index_dir}. "
                    "Run 'agentic-rag ingest' and 'agentic-rag index' first."
                )

            _retriever = Retriever.load(
                index_dir,
                get_embedding_provider(settings.embedding.provider, settings),
                rrf_k=settings.retrieval.rrf_k,
                candidate_pool=settings.retrieval.candidate_pool,
            )

        app.state.retriever = _retriever

        yield

        # Shutdown: nothing to do

    app = FastAPI(title="agentic-rag", lifespan=lifespan)

    # Exception handlers (register before routes)
    from fastapi.exceptions import RequestValidationError as _RVE

    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(_RVE, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, general_exception_handler)

    # Rate limiting
    def get_token_from_header(request: Any) -> str:
        """Extract token from Authorization header for rate limit key."""
        auth: str = request.headers.get("authorization", "")
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        addr: str = str(get_remote_address(request))
        return addr

    limiter = Limiter(key_func=get_token_from_header)
    app.state.limiter = limiter

    require_auth = bearer_auth(settings.api.token)

    # Routes. Handlers stay thin: validate, build library objects, delegate.
    # slowapi requires the decorated handler to accept a `request` parameter.

    @app.get("/health", tags=["system"])
    async def health() -> HealthResponse:
        """Health check endpoint (no authentication required)."""
        return HealthResponse(
            status="ok",
            index_loaded=_retriever is not None,
            provider=settings.provider,
            version=agentic_rag.__version__,
        )

    # response_model=None: the return union (JSON model | SSE stream) is not a
    # response-model FastAPI can infer; the JSON shape is still AskResponse.
    @app.post("/ask", tags=["chat"], response_model=None)
    @limiter.limit(settings.api.rate_limit)
    async def ask(
        request: Request,
        body: AskRequest,
        token: Annotated[str, Depends(require_auth)],
    ) -> AskResponse | StreamingResponse:
        """Answer a question from the corpus.

        Guardrails are always on — only the eval runner bypasses them, never
        the API. The agentic pipeline does not stream (the critic gates the
        final answer). On the streaming path, deltas are emitted before the
        output guardrail scan (same documented limitation as the CLI,
        ADR-008); the terminal `result` event carries the scanned final
        answer. Refusals are 200 responses with `refusal_reason` set, not
        errors.
        """
        if _retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")

        if body.pipeline == "agentic" and body.stream:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Agentic pipeline does not support streaming (critic gates the final answer)"
                ),
            )

        # Resolve provider
        provider_name = body.provider or settings.provider
        try:
            llm = get_llm_provider(provider_name, settings)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        # Fresh per request: Reranker.last_usage is call-scoped mutable state,
        # so a shared reranker across concurrent requests would misattribute
        # usage. Only the retriever (heavy: FAISS + BM25) is shared.
        rerank_mode = body.rerank or settings.rerank.mode
        reranker: Reranker
        if rerank_mode == "none":
            reranker = NoopReranker()
        elif rerank_mode == "llm":
            reranker = LLMReranker(llm, model=settings.rerank.model)
        elif rerank_mode == "cross-encoder":
            reranker = CrossEncoderReranker(model=settings.rerank.model)
        else:
            raise HTTPException(
                status_code=422,
                detail="Invalid rerank mode. Valid options: none, llm, cross-encoder",
            )

        pipeline: RAGPipeline | AgenticPipeline
        if body.pipeline == "agentic":
            pipeline = AgenticPipeline(_retriever, reranker, llm, settings)
        else:
            pipeline = RAGPipeline(_retriever, reranker, llm, settings)

        guarded = GuardedPipeline(
            pipeline,
            settings,
            provider=llm.name,
            model=provider_model(llm.name, settings),
            source="api",
        )

        mode = RetrievalMode(body.mode)

        if body.stream:
            return StreamingResponse(
                _stream_answer(guarded, body, mode, llm.name, reranker.name),
                media_type="text/event-stream",
            )

        try:
            result = await guarded.ask(body.question, mode=mode)
        except ProviderError as e:
            raise HTTPException(status_code=502, detail=f"Provider error: {e}") from e
        return _guarded_result_to_response(result, body, llm.name, reranker.name)

    @app.get("/search", tags=["search"])
    @limiter.limit(settings.api.rate_limit)
    async def search(
        request: Request,
        q: Annotated[str, Query(min_length=1, max_length=1000)],
        token: Annotated[str, Depends(require_auth)],
        mode: Annotated[str, Query(pattern="^(bm25|dense|hybrid)$")] = "hybrid",
        top_k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[SearchResult]:
        """Search the indexed corpus; returns ranked chunks (no LLM involved)."""
        if _retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")

        results = await _retriever.retrieve(q, mode=RetrievalMode(mode), top_k=top_k)
        return [
            SearchResult(
                rank=r.rank,
                score=r.score,
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                section_id=r.chunk.section_id,
                heading=r.chunk.heading,
                page_start=r.chunk.page_start,
                text=r.chunk.text[:300],
            )
            for r in results
        ]

    @app.get("/stats", tags=["observability"])
    @limiter.limit(settings.api.rate_limit)
    async def stats(
        request: Request,
        token: Annotated[str, Depends(require_auth)],
        by: Annotated[str, Query(pattern="^(provider|model|day|source|pipeline)$")] = "provider",
        since: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
        stages: bool = False,
    ) -> StatsResponse:
        """Aggregated metrics from the request ledger (see `agentic-rag stats`)."""
        # Check before constructing: MetricsStore.__init__ creates the db file.
        db_path = settings.metrics.db_path or (settings.data_dir / "metrics.db")
        if not settings.metrics.enabled or not db_path.exists():
            raise HTTPException(
                status_code=404, detail="Metrics store is disabled or has no data yet"
            )
        store = MetricsStore(db_path)

        summary = store.summary(group=by, since=since)
        stage_rows = store.stage_summary(since=since) if stages else None

        return StatsResponse(
            summary=[StatsRow(**cast(dict[str, Any], row)) for row in summary],
            stages=(
                [StageRow(**cast(dict[str, Any], row)) for row in stage_rows]
                if stage_rows is not None
                else None
            ),
        )

    return app


async def _stream_answer(
    guarded: GuardedPipeline,
    body: AskRequest,
    mode: RetrievalMode,
    provider: str,
    rerank: str,
) -> Any:
    """Stream answer as SSE: delta events, then one terminal result event."""
    try:
        async for event in guarded.ask_stream(body.question, mode=mode):
            if event.result is not None:
                response = _guarded_result_to_response(event.result, body, provider, rerank)
                yield f"event: result\ndata: {json.dumps(response.model_dump(mode='json'))}\n\n"
            elif event.delta:
                yield f"event: delta\ndata: {json.dumps({'delta': event.delta})}\n\n"
    except Exception:
        logger.exception("Error in stream")
        yield f"event: error\ndata: {json.dumps({'error': 'Streaming error'})}\n\n"


def _guarded_result_to_response(
    result: GuardedResult, body: AskRequest, provider: str, rerank: str
) -> AskResponse:
    """Convert GuardedResult to AskResponse (mirrors CLI._record + _guardrails_json).

    ``rerank`` is the name of the reranker actually used (the request may have
    left it None to inherit the configured default).
    """
    answer = result.answer

    citations = [
        CitationOut(
            marker=c.marker,
            chunk_id=c.chunk.chunk_id,
            doc_id=c.chunk.doc_id,
            section_id=c.chunk.section_id,
            heading=c.chunk.heading,
            page_start=c.chunk.page_start,
        )
        for c in answer.citations
    ]

    # The requested pipeline, not agent presence: a blocked input on an
    # agentic request returns no agent metadata but is still agentic.
    pipeline = body.pipeline

    # Build guardrails dict
    guardrails = GuardrailsOut(
        input=GuardrailScanOut(
            detections=[
                {
                    "detector": d.detection.detector,
                    "entity": d.detection.entity,
                    "action": d.action,
                }
                for d in result.input_verdict.applied
            ],
            blocked=result.input_verdict.blocked,
        ),
        output=(
            GuardrailScanOut(
                detections=[
                    {
                        "detector": d.detection.detector,
                        "entity": d.detection.entity,
                        "action": d.action,
                    }
                    for d in result.output_verdict.applied
                ],
                blocked=result.output_verdict.blocked,
            )
            if result.output_verdict
            else None
        ),
        retrieved_flagged_chunk_ids=list(result.retrieved_flagged_chunk_ids),
    )

    # Build agent dict if present
    agent_dict: dict[str, object] | None = None
    if result.agent is not None:
        agent_dict = {
            "plan": result.agent.plan.kind.value,
            "sub_queries": list(result.agent.plan.sub_queries),
            "revisions": result.agent.revisions,
            "caveat": result.agent.caveat,
        }

    return AskResponse(
        question=body.question,
        provider=provider,
        mode=body.mode,
        rerank=rerank,
        pipeline=pipeline,
        answer=answer.text,
        refusal=answer.refusal,
        refusal_reason=answer.refusal_reason,
        citations=citations,
        invalid_citations=list(answer.invalid_citations),
        usage=UsageOut(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            cost_usd=answer.usage.cost_usd,
        ),
        timings={t.stage: t.seconds for t in answer.timings},
        request_id=result.request_id,
        guardrails=guardrails,
        agent=agent_dict,
    )
