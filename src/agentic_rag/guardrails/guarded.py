"""Safety-sandwich wrapper: input scan pre-pipeline, output scan post-pipeline.

The GuardedPipeline wraps either a RAGPipeline (vanilla) or AgenticPipeline
(agentic), applying guardrails at three points:

1. INPUT: Scan the original question for PII and injection; apply policy.
   - If blocked: refuse immediately, write audit, return without running pipeline.
   - If redacted: pass redacted question to inner pipeline.

2. PIPELINE: Run inner.ask() or inner.ask_stream() on (possibly redacted) input.

3. RETRIEVED: For each context chunk, scan for injection patterns (audit-only,
   never mutate).

4. OUTPUT: Scan generated answer text for PII; apply policy.
   - If blocked: replace answer with refusal, write audit.
   - If redacted: apply redactions to final answer text.

5. AUDIT: Write one audit_v1 record per request, capturing input/output verdicts,
   detections, usage, latency, refusal reason, and answer hash. When audit is
   disabled, audit_path is None and no files are written.

Streaming limitation: deltas are passed through unscanned (they've already been
generated). Output redaction/blocking applies only to the final Answer, so if
the output verdict acts (redact or block), a stderr notice is printed to warn
that streamed content differs from stored content.

Design: eval/benchmarking paths construct pipelines directly and bypass this
wrapper (via --no-guardrails CLI flag), ensuring benchmarks measure the
underlying pipeline, not guardrail overhead.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from agentic_rag.agent.graph import AgenticPipeline
from agentic_rag.agent.state import AgentAnswer
from agentic_rag.config import Settings
from agentic_rag.guardrails.audit import AuditRecord, AuditWriter, ScanSummary, sha256_hex
from agentic_rag.guardrails.base import RefusalReason, Verdict
from agentic_rag.guardrails.injection import InjectionScanner
from agentic_rag.guardrails.pii import PIIScanner
from agentic_rag.guardrails.policy import apply_policy, load_policy
from agentic_rag.guardrails.refusal import render_refusal
from agentic_rag.observability import set_usage_attributes, tracer
from agentic_rag.observability.metrics import RequestMetric, metrics_store_for
from agentic_rag.pipeline.base import Answer, CitedChunk, StageTiming
from agentic_rag.pipeline.pipeline import RAGPipeline
from agentic_rag.providers.base import Usage
from agentic_rag.retrieval.base import RetrievalMode


@dataclasses.dataclass(frozen=True, slots=True)
class GuardedResult:
    """Result of a guarded ask() call.

    ``answer`` is the final Answer after all guardrail processing. ``agent``
    is set when the inner pipeline is agentic; its answer field is replaced
    with the final guardrail-processed answer. ``input_verdict`` is the policy
    verdict for the original question; ``output_verdict`` is None when input
    was blocked (pipeline never ran). ``retrieved_flagged_chunk_ids`` collects
    chunk IDs with injection detections (audit-only, text never mutated).
    ``audit_path`` is the file written by the audit writer, or None when
    auditing is disabled.
    """

    answer: Answer
    agent: AgentAnswer | None
    request_id: str
    input_verdict: Verdict
    output_verdict: Verdict | None
    retrieved_flagged_chunk_ids: tuple[str, ...]
    audit_path: Path | None


@dataclasses.dataclass(frozen=True, slots=True)
class GuardedStreamEvent:
    """Event from streaming guarded ask_stream() call.

    Text events carry ``delta``; the single terminal event carries the final
    ``result`` (GuardedResult with guardrails applied) and an empty delta.
    """

    delta: str = ""
    result: GuardedResult | None = None


def provider_model(provider: str, settings: Settings) -> str:
    """Resolve the model string for audit records.

    Mirrors evals/generation._get_provider_model without importing from evals
    (layering constraint). Returns the configured model name for the provider.
    """
    match provider:
        case "anthropic":
            config = settings.anthropic
            if config.backend == "bedrock" and config.bedrock_model:
                return config.bedrock_model
            return config.model
        case "openai":
            return settings.openai.model
        case "google":
            return settings.google.model
        case "ollama":
            return settings.ollama.model
        case _:
            return "unknown"


class GuardedPipeline:
    """Wraps RAGPipeline or AgenticPipeline with input/output guardrails.

    Applies the safety sandwich: scan input, run pipeline, scan retrieved
    chunks and output, apply policy, audit, and return a GuardedResult with
    both the processed answer and the inner AgentAnswer (if agentic).
    """

    def __init__(
        self,
        inner: RAGPipeline | AgenticPipeline,
        settings: Settings,
        *,
        provider: str,
        model: str,
        source: str = "cli",
    ) -> None:
        """Initialize the guarded pipeline.

        Args:
            inner: RAGPipeline or AgenticPipeline to wrap.
            settings: Application settings (guardrails, audit dirs, etc).
            provider: Provider name for audit records (e.g., "anthropic").
            model: Model identifier for audit records.
            source: Metrics source identifier (default "cli").
        """
        self.inner = inner
        self.settings = settings
        self.provider = provider
        self.model = model
        self.source = source

        self.policy = load_policy(settings.guardrails.policy_file)
        self.pii = PIIScanner(ner=settings.guardrails.ner)
        self.injection = InjectionScanner()

        audit_dir = settings.guardrails.audit_dir or settings.data_dir / "audit"
        self.audit_writer = AuditWriter(audit_dir) if settings.guardrails.audit_enabled else None
        self.metrics = metrics_store_for(settings)

    async def ask(
        self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> GuardedResult:
        """Answer a question with input/output guardrails and audit logging.

        Flow:
        1. INPUT: Scan question for PII/injection; block or redact per policy.
        2. PIPELINE: Run inner.ask() on (possibly redacted) question.
        3. RETRIEVED: Scan each context chunk for injection (audit-only).
        4. OUTPUT: Scan answer text for PII; block or redact per policy.
        5. AUDIT: Write one record capturing all verdicts and audit data.

        Args:
            question: The user's question (original, pre-scan).
            mode: Retrieval mode (BM25, dense, or hybrid).

        Returns:
            GuardedResult with final answer, verdicts, and audit path.
        """
        request_id = uuid.uuid4().hex
        ts = datetime.now(UTC).isoformat()
        query_sha256 = sha256_hex(question)
        raw_query = question if self.settings.guardrails.log_raw_query else None

        with tracer().start_as_current_span("rag.request") as root_span:
            root_span.set_attribute("rag.request_id", request_id)
            root_span.set_attribute("rag.provider", self.provider)
            root_span.set_attribute("rag.model", self.model)
            root_span.set_attribute(
                "rag.pipeline", "agentic" if isinstance(self.inner, AgenticPipeline) else "vanilla"
            )
            root_span.set_attribute("rag.mode", mode.value)
            root_span.set_attribute("rag.rerank", self.inner.settings.rerank.mode)

            t_in_start = time.perf_counter()

            # INPUT: Scan original question
            with tracer().start_as_current_span("guardrails.input") as input_span:
                input_detections = self.pii.scan(question) + self.injection.scan(question)
                input_verdict = apply_policy(
                    self.policy, question, input_detections, direction="input"
                )

                input_redactions = sum(
                    1 for a in input_verdict.applied if a.action.value == "redact"
                )
                input_span.set_attribute("guardrails.detections", len(input_detections))
                input_span.set_attribute("guardrails.blocked", input_verdict.blocked)
                input_span.set_attribute("guardrails.redactions", input_redactions)

            elapsed_in = time.perf_counter() - t_in_start

            # If input blocked, refuse immediately
            if input_verdict.blocked:
                reason = RefusalReason.INPUT_PII
                for det in input_verdict.applied:
                    if det.detection.detector == "injection":
                        reason = RefusalReason.INPUT_INJECTION
                        break

                text = render_refusal(self.policy, reason, input_verdict.applied)
                answer = Answer(
                    text=text,
                    citations=[],
                    context=[],
                    usage=Usage.zero(),
                    timings=[StageTiming("guardrails_in", elapsed_in)],
                    refusal=True,
                    invalid_citations=[],
                    refusal_reason=reason.value,
                )

                # Write audit without answer/output
                audit_record = AuditRecord(
                    request_id=request_id,
                    ts=ts,
                    query_sha256=query_sha256,
                    raw_query=raw_query,
                    provider=self.provider,
                    model=self.model,
                    pipeline="agentic" if isinstance(self.inner, AgenticPipeline) else "vanilla",
                    mode=mode.value,
                    rerank=self.inner.settings.rerank.mode,
                    guardrails_enabled=True,
                    policy_version=self.policy.version,
                    ner=self.settings.guardrails.ner,
                    input_scan=ScanSummary(input_verdict.applied, input_verdict.blocked),
                    output_scan=None,
                    retrieved_flagged_chunk_ids=(),
                    chunk_ids=(),
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    latency_s={"guardrails_in": elapsed_in, "total": elapsed_in},
                    refusal=True,
                    refusal_reason=reason.value,
                    answer_sha256=None,
                )
                audit_path = self.audit_writer.write(audit_record) if self.audit_writer else None

                # Record metrics (don't let failures break requests)
                if self.metrics is not None:
                    try:
                        metric = RequestMetric(
                            request_id=request_id,
                            ts=ts,
                            source=self.source,
                            provider=self.provider,
                            model=self.model,
                            pipeline="agentic"
                            if isinstance(self.inner, AgenticPipeline)
                            else "vanilla",
                            mode=mode.value,
                            rerank=self.inner.settings.rerank.mode,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=None,
                            latency_s=elapsed_in,
                            stages={"guardrails_in": elapsed_in},
                            refusal=True,
                            refusal_reason=reason.value,
                        )
                        self.metrics.record(metric)
                    except Exception as exc:
                        print(
                            f"metrics record failed: {exc}",
                            file=sys.stderr,
                        )

                root_span.set_attribute("rag.refusal", True)
                root_span.set_attribute("rag.refusal_reason", reason.value)
                root_span.set_attribute("guardrails.retrieved_flagged_count", 0)

                return GuardedResult(
                    answer=answer,
                    agent=None,
                    request_id=request_id,
                    input_verdict=input_verdict,
                    output_verdict=None,
                    retrieved_flagged_chunk_ids=(),
                    audit_path=audit_path,
                )

            # PIPELINE: Run inner on (possibly redacted) question
            result = await self.inner.ask(input_verdict.text, mode=mode)

            # Unwrap agent result if present
            if isinstance(result, AgentAnswer):
                answer = result.answer
                agent_meta = result
            else:
                answer = result
                agent_meta = None

            # RETRIEVED: Scan each context chunk for injection (audit-only)
            retrieved_flagged_chunk_ids: list[str] = []
            for scored in answer.context:
                injection_dets = self.injection.scan(scored.chunk.text)
                if injection_dets:
                    _ = apply_policy(
                        self.policy, scored.chunk.text, injection_dets, direction="retrieved"
                    )
                    retrieved_flagged_chunk_ids.append(scored.chunk.chunk_id)

            # OUTPUT: Scan answer text for PII
            t_out_start = time.perf_counter()
            with tracer().start_as_current_span("guardrails.output") as output_span:
                output_detections = self.pii.scan(answer.text)
                output_verdict = apply_policy(
                    self.policy, answer.text, output_detections, direction="output"
                )
                output_redactions = sum(
                    1 for a in output_verdict.applied if a.action.value == "redact"
                )
                output_span.set_attribute("guardrails.detections", len(output_detections))
                output_span.set_attribute("guardrails.blocked", output_verdict.blocked)
                output_span.set_attribute("guardrails.redactions", output_redactions)

            elapsed_out = time.perf_counter() - t_out_start

            # Determine final answer after output policy
            if output_verdict.blocked:
                # Output blocked: replace with refusal
                final_text = render_refusal(
                    self.policy, RefusalReason.OUTPUT_PII, output_verdict.applied
                )
                refusal = True
                refusal_reason: str | None = RefusalReason.OUTPUT_PII.value
                final_citations: list[CitedChunk] = []
            else:
                # Output not blocked: use redacted text
                final_text = output_verdict.text
                refusal = answer.refusal
                refusal_reason = RefusalReason.OUT_OF_CORPUS.value if answer.refusal else None
                final_citations = answer.citations

            # Build final answer
            final_answer = dataclasses.replace(
                answer,
                text=final_text,
                refusal=refusal,
                refusal_reason=refusal_reason,
                citations=final_citations,
                timings=[
                    *answer.timings,
                    StageTiming("guardrails_in", elapsed_in),
                    StageTiming("guardrails_out", elapsed_out),
                ],
            )

            # Update agent wrapper if present
            if agent_meta is not None:
                agent_out = dataclasses.replace(agent_meta, answer=final_answer)
            else:
                agent_out = None

            # AUDIT: Write record
            answer_sha256 = (
                sha256_hex(final_answer.text)
                if final_answer.text and not input_verdict.blocked
                else None
            )

            latency_timings: dict[str, float] = {t.stage: t.seconds for t in final_answer.timings}
            latency_timings["total"] = sum(t.seconds for t in final_answer.timings)

            audit_record = AuditRecord(
                request_id=request_id,
                ts=ts,
                query_sha256=query_sha256,
                raw_query=raw_query,
                provider=self.provider,
                model=self.model,
                pipeline="agentic" if agent_meta else "vanilla",
                mode=mode.value,
                rerank=self.inner.settings.rerank.mode,
                guardrails_enabled=True,
                policy_version=self.policy.version,
                ner=self.settings.guardrails.ner,
                input_scan=ScanSummary(input_verdict.applied, input_verdict.blocked),
                output_scan=ScanSummary(output_verdict.applied, output_verdict.blocked),
                retrieved_flagged_chunk_ids=tuple(retrieved_flagged_chunk_ids),
                chunk_ids=tuple(s.chunk.chunk_id for s in final_answer.context),
                input_tokens=final_answer.usage.input_tokens,
                output_tokens=final_answer.usage.output_tokens,
                cost_usd=final_answer.usage.cost_usd,
                latency_s=latency_timings,
                refusal=final_answer.refusal,
                refusal_reason=final_answer.refusal_reason,
                answer_sha256=answer_sha256,
            )
            audit_path = self.audit_writer.write(audit_record) if self.audit_writer else None

            # Record metrics (don't let failures break requests)
            if self.metrics is not None:
                try:
                    # stages = latency_timings minus "total"
                    stages = {k: v for k, v in latency_timings.items() if k != "total"}
                    metric = RequestMetric(
                        request_id=request_id,
                        ts=ts,
                        source=self.source,
                        provider=self.provider,
                        model=self.model,
                        pipeline="agentic" if agent_meta else "vanilla",
                        mode=mode.value,
                        rerank=self.inner.settings.rerank.mode,
                        input_tokens=final_answer.usage.input_tokens,
                        output_tokens=final_answer.usage.output_tokens,
                        cost_usd=final_answer.usage.cost_usd,
                        latency_s=latency_timings["total"],
                        stages=stages,
                        refusal=final_answer.refusal,
                        refusal_reason=final_answer.refusal_reason,
                    )
                    self.metrics.record(metric)
                except Exception as exc:
                    print(
                        f"metrics record failed: {exc}",
                        file=sys.stderr,
                    )

            # Set final root span attributes
            root_span.set_attribute("rag.refusal", final_answer.refusal)
            if final_answer.refusal_reason is not None:
                root_span.set_attribute("rag.refusal_reason", final_answer.refusal_reason)
            set_usage_attributes(root_span, final_answer.usage)
            root_span.set_attribute(
                "guardrails.retrieved_flagged_count", len(retrieved_flagged_chunk_ids)
            )

            return GuardedResult(
                answer=final_answer,
                agent=agent_out,
                request_id=request_id,
                input_verdict=input_verdict,
                output_verdict=output_verdict,
                retrieved_flagged_chunk_ids=tuple(retrieved_flagged_chunk_ids),
                audit_path=audit_path,
            )

    def ask_stream(
        self, question: str, *, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> AsyncIterator[GuardedStreamEvent]:
        """Stream answer with input scan before, output scan after streaming.

        Note: Streaming content is not scanned (it's already generated). Only
        the final Answer text is scanned for PII/injection and redacted/blocked.
        If the output verdict acts, a stderr warning is printed.

        Args:
            question: The user's question.
            mode: Retrieval mode.

        Yields:
            GuardedStreamEvent with delta (text) or result (terminal).

        Raises:
            TypeError: If inner doesn't support streaming (e.g., AgenticPipeline).
        """
        if not hasattr(self.inner, "ask_stream") or isinstance(self.inner, AgenticPipeline):
            raise TypeError(
                "ask_stream not supported (agentic pipeline's critic gates the final answer)"
            )

        async def _stream() -> AsyncIterator[GuardedStreamEvent]:
            request_id = uuid.uuid4().hex
            ts = datetime.now(UTC).isoformat()
            query_sha256 = sha256_hex(question)
            raw_query = question if self.settings.guardrails.log_raw_query else None

            # Start root span for the entire streaming operation
            with tracer().start_as_current_span("rag.request") as root_span:
                root_span.set_attribute("rag.request_id", request_id)
                root_span.set_attribute("rag.provider", self.provider)
                root_span.set_attribute("rag.model", self.model)
                root_span.set_attribute("rag.pipeline", "vanilla")
                root_span.set_attribute("rag.mode", mode.value)
                root_span.set_attribute("rag.rerank", self.inner.settings.rerank.mode)

                t_in_start = time.perf_counter()

                # INPUT: Scan original question
                with tracer().start_as_current_span("guardrails.input") as input_span:
                    input_detections = self.pii.scan(question) + self.injection.scan(question)
                    input_verdict = apply_policy(
                        self.policy, question, input_detections, direction="input"
                    )

                    input_redactions = sum(
                        1 for a in input_verdict.applied if a.action.value == "redact"
                    )
                    input_span.set_attribute("guardrails.detections", len(input_detections))
                    input_span.set_attribute("guardrails.blocked", input_verdict.blocked)
                    input_span.set_attribute("guardrails.redactions", input_redactions)

                elapsed_in = time.perf_counter() - t_in_start

                # If input blocked, yield one terminal refusal event
                if input_verdict.blocked:
                    reason = RefusalReason.INPUT_PII
                    for det in input_verdict.applied:
                        if det.detection.detector == "injection":
                            reason = RefusalReason.INPUT_INJECTION
                            break

                    text = render_refusal(self.policy, reason, input_verdict.applied)
                    answer = Answer(
                        text=text,
                        citations=[],
                        context=[],
                        usage=Usage.zero(),
                        timings=[StageTiming("guardrails_in", elapsed_in)],
                        refusal=True,
                        invalid_citations=[],
                        refusal_reason=reason.value,
                    )

                    # Write audit without answer/output
                    audit_record = AuditRecord(
                        request_id=request_id,
                        ts=ts,
                        query_sha256=query_sha256,
                        raw_query=raw_query,
                        provider=self.provider,
                        model=self.model,
                        pipeline="vanilla",
                        mode=mode.value,
                        rerank=self.inner.settings.rerank.mode,
                        guardrails_enabled=True,
                        policy_version=self.policy.version,
                        ner=self.settings.guardrails.ner,
                        input_scan=ScanSummary(input_verdict.applied, input_verdict.blocked),
                        output_scan=None,
                        retrieved_flagged_chunk_ids=(),
                        chunk_ids=(),
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=None,
                        latency_s={"guardrails_in": elapsed_in, "total": elapsed_in},
                        refusal=True,
                        refusal_reason=reason.value,
                        answer_sha256=None,
                    )
                    audit_path = (
                        self.audit_writer.write(audit_record) if self.audit_writer else None
                    )

                    # Record metrics (don't let failures break requests)
                    if self.metrics is not None:
                        try:
                            metric = RequestMetric(
                                request_id=request_id,
                                ts=ts,
                                source=self.source,
                                provider=self.provider,
                                model=self.model,
                                pipeline="vanilla",
                                mode=mode.value,
                                rerank=self.inner.settings.rerank.mode,
                                input_tokens=0,
                                output_tokens=0,
                                cost_usd=None,
                                latency_s=elapsed_in,
                                stages={"guardrails_in": elapsed_in},
                                refusal=True,
                                refusal_reason=reason.value,
                            )
                            self.metrics.record(metric)
                        except Exception as exc:
                            print(
                                f"metrics record failed: {exc}",
                                file=sys.stderr,
                            )

                    root_span.set_attribute("rag.refusal", True)
                    root_span.set_attribute("rag.refusal_reason", reason.value)
                    root_span.set_attribute("guardrails.retrieved_flagged_count", 0)

                    yield GuardedStreamEvent(
                        result=GuardedResult(
                            answer=answer,
                            agent=None,
                            request_id=request_id,
                            input_verdict=input_verdict,
                            output_verdict=None,
                            retrieved_flagged_chunk_ids=(),
                            audit_path=audit_path,
                        )
                    )
                    return

                # PIPELINE: Stream from inner on (possibly redacted) question
                t_out_start = time.perf_counter()
                final_answer: Answer | None = None
                async for event in cast(RAGPipeline, self.inner).ask_stream(
                    input_verdict.text, mode=mode
                ):
                    if event.answer is not None:
                        # Terminal event: process answer through output guardrails
                        final_answer = event.answer

                        # RETRIEVED: Scan each context chunk for injection (audit-only)
                        retrieved_flagged_chunk_ids: list[str] = []
                        for scored in final_answer.context:
                            injection_dets = self.injection.scan(scored.chunk.text)
                            if injection_dets:
                                _ = apply_policy(
                                    self.policy,
                                    scored.chunk.text,
                                    injection_dets,
                                    direction="retrieved",
                                )
                                retrieved_flagged_chunk_ids.append(scored.chunk.chunk_id)

                        # OUTPUT: Scan answer text for PII
                        with tracer().start_as_current_span("guardrails.output") as output_span:
                            output_detections = self.pii.scan(final_answer.text)
                            output_verdict = apply_policy(
                                self.policy,
                                final_answer.text,
                                output_detections,
                                direction="output",
                            )

                            output_redactions = sum(
                                1 for a in output_verdict.applied if a.action.value == "redact"
                            )
                            output_span.set_attribute(
                                "guardrails.detections", len(output_detections)
                            )
                            output_span.set_attribute("guardrails.blocked", output_verdict.blocked)
                            output_span.set_attribute("guardrails.redactions", output_redactions)

                        elapsed_out = time.perf_counter() - t_out_start

                        # Determine final answer after output policy
                        if output_verdict.blocked:
                            final_text = render_refusal(
                                self.policy, RefusalReason.OUTPUT_PII, output_verdict.applied
                            )
                            refusal = True
                            refusal_reason: str | None = RefusalReason.OUTPUT_PII.value
                            final_citations: list[CitedChunk] = []
                        else:
                            final_text = output_verdict.text
                            refusal = final_answer.refusal
                            refusal_reason = (
                                RefusalReason.OUT_OF_CORPUS.value if final_answer.refusal else None
                            )
                            final_citations = final_answer.citations

                        # Build final answer
                        processed_answer = dataclasses.replace(
                            final_answer,
                            text=final_text,
                            refusal=refusal,
                            refusal_reason=refusal_reason,
                            citations=final_citations,
                            timings=[
                                *final_answer.timings,
                                StageTiming("guardrails_in", elapsed_in),
                                StageTiming("guardrails_out", elapsed_out),
                            ],
                        )

                        # AUDIT: Write record
                        answer_sha256 = (
                            sha256_hex(processed_answer.text) if processed_answer.text else None
                        )

                        latency_timings: dict[str, float] = {
                            t.stage: t.seconds for t in processed_answer.timings
                        }
                        latency_timings["total"] = sum(t.seconds for t in processed_answer.timings)

                        audit_record = AuditRecord(
                            request_id=request_id,
                            ts=ts,
                            query_sha256=query_sha256,
                            raw_query=raw_query,
                            provider=self.provider,
                            model=self.model,
                            pipeline="vanilla",
                            mode=mode.value,
                            rerank=self.inner.settings.rerank.mode,
                            guardrails_enabled=True,
                            policy_version=self.policy.version,
                            ner=self.settings.guardrails.ner,
                            input_scan=ScanSummary(input_verdict.applied, input_verdict.blocked),
                            output_scan=ScanSummary(output_verdict.applied, output_verdict.blocked),
                            retrieved_flagged_chunk_ids=tuple(retrieved_flagged_chunk_ids),
                            chunk_ids=tuple(s.chunk.chunk_id for s in processed_answer.context),
                            input_tokens=processed_answer.usage.input_tokens,
                            output_tokens=processed_answer.usage.output_tokens,
                            cost_usd=processed_answer.usage.cost_usd,
                            latency_s=latency_timings,
                            refusal=processed_answer.refusal,
                            refusal_reason=processed_answer.refusal_reason,
                            answer_sha256=answer_sha256,
                        )
                        audit_path = (
                            self.audit_writer.write(audit_record) if self.audit_writer else None
                        )

                        # Record metrics (don't let failures break requests)
                        if self.metrics is not None:
                            try:
                                # stages = latency_timings minus "total"
                                stages = {k: v for k, v in latency_timings.items() if k != "total"}
                                metric = RequestMetric(
                                    request_id=request_id,
                                    ts=ts,
                                    source=self.source,
                                    provider=self.provider,
                                    model=self.model,
                                    pipeline="vanilla",
                                    mode=mode.value,
                                    rerank=self.inner.settings.rerank.mode,
                                    input_tokens=processed_answer.usage.input_tokens,
                                    output_tokens=processed_answer.usage.output_tokens,
                                    cost_usd=processed_answer.usage.cost_usd,
                                    latency_s=latency_timings["total"],
                                    stages=stages,
                                    refusal=processed_answer.refusal,
                                    refusal_reason=processed_answer.refusal_reason,
                                )
                                self.metrics.record(metric)
                            except Exception as exc:
                                print(
                                    f"metrics record failed: {exc}",
                                    file=sys.stderr,
                                )

                        # Set final root span attributes
                        root_span.set_attribute("rag.refusal", processed_answer.refusal)
                        if processed_answer.refusal_reason is not None:
                            root_span.set_attribute(
                                "rag.refusal_reason", processed_answer.refusal_reason
                            )
                        set_usage_attributes(root_span, processed_answer.usage)
                        root_span.set_attribute(
                            "guardrails.retrieved_flagged_count", len(retrieved_flagged_chunk_ids)
                        )

                        yield GuardedStreamEvent(
                            result=GuardedResult(
                                answer=processed_answer,
                                agent=None,
                                request_id=request_id,
                                input_verdict=input_verdict,
                                output_verdict=output_verdict,
                                retrieved_flagged_chunk_ids=tuple(retrieved_flagged_chunk_ids),
                                audit_path=audit_path,
                            )
                        )
                    else:
                        # Pass-through delta events
                        yield GuardedStreamEvent(delta=event.delta)

        return _stream()
