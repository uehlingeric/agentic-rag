"""Draft synthesis with optional revision via critique feedback.

Generates answers from context via LLM, with optional revision loop that
incorporates critique issues. Handles sentinel detection for refusals (leading
[NO_ANSWER]) and the week-4 failure mode where a partial answer is followed by
[NO_ANSWER] (trailing sentinel).
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.state import Critique
from agentic_rag.pipeline.base import NO_ANSWER_SENTINEL
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, Usage


@dataclass(frozen=True, slots=True)
class DraftResult:
    """One synthesis pass.

    ``text`` is sentinel-stripped. ``trailing_sentinel`` records the week-4
    failure mode (model appends [NO_ANSWER] after a partial answer): the
    trailing sentinel is stripped WITHOUT marking refusal.
    """

    text: str
    refusal: bool
    trailing_sentinel: bool
    usage: Usage
    model: str
    prompt_id: str


async def synthesize_draft(
    llm: LLMProvider,
    question: str,
    context: BuiltContext,
    *,
    prior_draft: str | None = None,
    critique: Critique | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    prompt_version: int | None = None,
) -> DraftResult:
    """Generate or revise a draft answer from context.

    First pass (prior_draft and critique both None): renders agent-synthesis
    prompt with context and question, sends as single USER message.

    Revision pass (both prior_draft and critique provided): renders
    agent-synthesis as before, adds ASSISTANT message with prior_draft, then
    adds USER message with agent-revise prompt containing critique issues
    formatted as ``- {kind}: {detail} Fix: {fix}`` (one per line).

    Args:
        llm: LLM provider for completion.
        question: The user's question.
        context: Built context from retrieval.
        prior_draft: Prior answer text (for revision); None for first pass.
        critique: Critique with issues (for revision); None for first pass.
        model: Model name to use; None uses provider default.
        max_tokens: Maximum output tokens.
        prompt_version: Agent-synthesis prompt version pin; agent-revise is
            loaded latest (not independently pinned).

    Returns:
        DraftResult with text, refusal/trailing_sentinel flags, usage, and
        model.

    Raises:
        ValueError: If prior_draft and critique are not both None or both
            provided.
    """
    # Validate prior_draft and critique consistency
    if (prior_draft is None) != (critique is None):
        raise ValueError("prior_draft and critique must both be None or both be provided")

    # Load and render agent-synthesis prompt
    synthesis_prompt = load_prompt("agent-synthesis", version=prompt_version)
    synthesis_text = synthesis_prompt.render(context=context.text, question=question)

    # Prepare messages based on pass type
    if prior_draft is None:
        # First pass: single USER message
        messages = [Message(role=Role.USER, content=synthesis_text)]
    else:
        # Revision pass: 3 messages
        # critique is guaranteed to be not None by the validation above
        assert critique is not None
        # Format critique issues as "- {kind}: {detail} Fix: {fix}" lines
        issue_lines = [
            f"- {issue.kind}: {issue.detail} Fix: {issue.fix}" for issue in critique.issues
        ]
        issues_text = "\n".join(issue_lines)

        # Load revise prompt and render it
        revise_prompt = load_prompt("agent-revise")
        revise_text = revise_prompt.render(issues=issues_text)

        messages = [
            Message(role=Role.USER, content=synthesis_text),
            Message(role=Role.ASSISTANT, content=prior_draft),
            Message(role=Role.USER, content=revise_text),
        ]

    # Call LLM with temperature 0.0
    completion = await llm.complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # Post-process: detect and strip sentinels
    text = completion.text.lstrip()

    # Detect leading sentinel (refusal)
    refusal = text.startswith(NO_ANSWER_SENTINEL)
    if refusal:
        text = text[len(NO_ANSWER_SENTINEL) :].lstrip()

    # Detect trailing sentinel (week-4 failure mode)
    trailing_sentinel = False
    text = text.rstrip()
    if text.endswith(NO_ANSWER_SENTINEL):
        trailing_sentinel = True
        text = text[: -len(NO_ANSWER_SENTINEL)].rstrip()

    return DraftResult(
        text=text,
        refusal=refusal,
        trailing_sentinel=trailing_sentinel,
        usage=completion.usage,
        model=completion.model,
        prompt_id=synthesis_prompt.id,
    )
