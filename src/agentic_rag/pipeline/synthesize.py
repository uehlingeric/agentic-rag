"""Synthesis: LLM-powered answer generation with refusal detection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL, scrub_sentinel
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, StreamEvent, Usage


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Result of synthesis (context + question -> answer).

    ``text`` has the sentinel stripped and leading/trailing whitespace trimmed.
    ``refusal`` is True when the model indicated the corpus cannot answer.
    ``stray_sentinel`` is True when a non-leading sentinel occurrence was removed
    mid-answer or trailing (the week-5 cross-provider failure mode); the text is
    kept and no refusal is marked.
    ``usage`` and ``model`` come from the LLM completion.
    """

    text: str
    refusal: bool
    stray_sentinel: bool
    usage: Usage
    model: str


async def synthesize(
    llm: LLMProvider,
    question: str,
    context: BuiltContext,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    prompt_version: int | None = None,
) -> SynthesisResult:
    """Generate an answer from context using an LLM.

    Loads the synthesis prompt, renders it with context and question, and calls
    the LLM. Detects refusals (model output starting with NO_ANSWER_SENTINEL)
    and strips the sentinel.

    Args:
        llm: LLM provider for completion.
        question: The user's question.
        context: Built context from retrieval.
        model: Model name to use; None uses provider default.
        max_tokens: Maximum output tokens.
        prompt_version: Synthesis prompt version; None uses latest.

    Returns:
        SynthesisResult with text, refusal flag, usage, and model.
    """
    prompt = load_prompt("synthesis", version=prompt_version)
    prompt_text = prompt.render(context=context.text, question=question)

    completion = await llm.complete(
        messages=[Message(role=Role.USER, content=prompt_text)],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    scrub = scrub_sentinel(completion.text)

    return SynthesisResult(
        text=scrub.text,
        refusal=scrub.refusal,
        stray_sentinel=scrub.stray_sentinel,
        usage=completion.usage,
        model=completion.model,
    )


async def stream_synthesis(
    llm: LLMProvider,
    question: str,
    context: BuiltContext,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    prompt_version: int | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream an answer, buffering the sentinel so it never reaches the consumer.

    Loads the synthesis prompt and wraps llm.stream with sentinel buffering.
    Maintains a buffer during the initial "undecided" phase. As deltas arrive:
    - If the stripped buffer is empty or could still be a sentinel prefix:
      buffer the delta, emit nothing.
    - Once the stripped buffer starts with the full sentinel:
      emit the portion after the sentinel, then pass through subsequent deltas.
    - Once the stripped buffer doesn't start with the sentinel:
      emit the original buffer (preserving leading whitespace), then pass through.

    The terminal event (with completion) is always forwarded unchanged; the
    completion.text still contains the raw sentinel and must be post-processed
    by the caller.

    If the stream ends while still undecided (total output shorter than sentinel
    prefix), emits the pending buffer as a final delta before forwarding the
    terminal event.

    Args:
        llm: LLM provider for streaming completion.
        question: The user's question.
        context: Built context from retrieval.
        model: Model name to use; None uses provider default.
        max_tokens: Maximum output tokens.
        prompt_version: Synthesis prompt version; None uses latest.

    Yields:
        StreamEvent objects with deltas; terminal event with completion.
    """
    prompt = load_prompt("synthesis", version=prompt_version)
    prompt_text = prompt.render(context=context.text, question=question)

    buffer = ""
    decided = False

    async for event in llm.stream(
        messages=[Message(role=Role.USER, content=prompt_text)],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        # Terminal event: always forward unchanged
        if event.completion is not None:
            # Emit pending buffer before forwarding terminal event
            if not decided and buffer:
                yield StreamEvent(delta=buffer)
            yield event
            return

        # Text delta event
        delta = event.delta
        if decided:
            # Already decided; pass through all deltas
            yield StreamEvent(delta=delta)
        else:
            # Still deciding; buffer and check
            buffer += delta
            s = buffer.lstrip()

            # Check if buffer could still be building the sentinel
            if s == "" or (len(s) < len(NO_ANSWER_SENTINEL) and NO_ANSWER_SENTINEL.startswith(s)):
                # Still undecided, keep buffering
                pass
            elif s.startswith(NO_ANSWER_SENTINEL):
                # Decided: this is a refusal
                decided = True
                # Emit everything after the sentinel (may be empty)
                remainder = s[len(NO_ANSWER_SENTINEL) :]
                if remainder:
                    yield StreamEvent(delta=remainder)
            else:
                # Decided: normal response (not a refusal)
                decided = True
                # Emit the original buffer, preserving leading whitespace
                yield StreamEvent(delta=buffer)
