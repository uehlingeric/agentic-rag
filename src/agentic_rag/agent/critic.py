"""LLM-as-critic for draft answer revision guidance (ADR-007, prompt-separated from the judge).

The critic improves answers; the judge scores them. The critic reads a draft
and yields specific, actionable revision guidance; it must not burn paid
revision cycles on garbage guidance, so it fails open to PASS (fallback=True)
when parse fails after all repair attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentic_rag.agent.state import CriticIssue, CriticVerdict, Critique, IssueKind
from agentic_rag.pipeline.context import BuiltContext
from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, Usage

_REPAIR_INSTRUCTION = (
    "Your previous reply was not the required JSON object. Reply with ONLY the JSON "
    'object in exactly the shape given in the instructions: either {"verdict": "pass"} '
    'or {"verdict": "revise", "issues": [list of issue objects]}, where each issue has '
    '"kind" (one of: uncited_claim, unsupported_citation, incomplete, contradiction), '
    'non-empty "detail" (what is wrong, quoting the offending text), and non-empty "fix" '
    "(the specific change to make). No code fences, no surrounding prose."
)


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """Critic call outcome. ``fallback`` is True when every parse attempt failed
    and the critique defaulted to PASS — a broken critic must not burn paid
    revision cycles on garbage guidance, so it fails open to vanilla behavior."""

    critique: Critique
    usage: Usage  # sums all attempts including repair turns
    prompt_id: str
    fallback: bool
    raw: str  # last model reply verbatim, for --trace


def _parse_critique(text: str) -> Critique:
    """Parse JSON critique reply. Raises ValueError on any validation failure."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in critic reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("critic reply is not a JSON object")

    verdict_str = data.get("verdict")
    if verdict_str not in ("pass", "revise"):
        raise ValueError(f"verdict must be 'pass' or 'revise', got {verdict_str!r}")

    if verdict_str == "pass":
        # Pass wins; ignore any stray issues
        return Critique(verdict=CriticVerdict.PASS, issues=())

    # Revise case: issues must be non-empty list
    issues_raw = data.get("issues")
    if not isinstance(issues_raw, list) or not issues_raw:
        raise ValueError("revise verdict requires non-empty issues list")

    parsed_issues: list[CriticIssue] = []
    for i, issue_data in enumerate(issues_raw):
        if not isinstance(issue_data, dict):
            raise ValueError(f"issue {i} is not a dict")
        kind_str = issue_data.get("kind")
        if not isinstance(kind_str, str):
            raise ValueError(f"issue {i}: kind must be a string, got {kind_str!r}")
        try:
            # Enum value lookup, not membership test: `str in EnumType` raises
            # TypeError on Python 3.11 (value-membership arrived in 3.12)
            kind = IssueKind(kind_str)
        except ValueError:
            raise ValueError(
                f"issue {i}: kind must be one of {[k.value for k in IssueKind]}, got {kind_str!r}"
            ) from None
        detail = issue_data.get("detail")
        fix = issue_data.get("fix")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"issue {i}: missing or empty detail")
        if not isinstance(fix, str) or not fix.strip():
            raise ValueError(f"issue {i}: missing or empty fix")
        parsed_issues.append(CriticIssue(kind=kind, detail=detail.strip(), fix=fix.strip()))

    return Critique(verdict=CriticVerdict.REVISE, issues=tuple(parsed_issues))


async def critique_draft(
    llm: LLMProvider,
    question: str,
    context: BuiltContext,
    draft: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    prompt_version: int | None = None,
    max_parse_retries: int = 2,
) -> CritiqueResult:
    """Critique a draft answer for revision guidance.

    Args:
        llm: LLM provider for the critique call.
        question: The original question the draft answers.
        context: Built context excerpts (BuiltContext.text passed to prompt).
        draft: The draft answer text to review.
        model: Optional model override (None uses provider default).
        max_tokens: Maximum tokens for the model response.
        prompt_version: Optional prompt version pin (None uses latest).
        max_parse_retries: Maximum JSON repair attempts before fail-open.

    Returns:
        CritiqueResult with critique (PASS or REVISE with issues), summed usage,
        prompt_id, fallback flag (True if parse failed on all attempts), and raw
        last reply for tracing.
    """
    prompt = load_prompt("critic", version=prompt_version)
    rendered = prompt.render(question=question, context=context.text, draft=draft)
    messages: list[Message] = [Message(role=Role.USER, content=rendered)]
    usage = Usage.zero()
    last_raw = ""

    for _ in range(max_parse_retries + 1):
        completion = await llm.complete(
            messages, model=model, max_tokens=max_tokens, temperature=0.0
        )
        usage = usage + completion.usage
        last_raw = completion.text
        try:
            critique = _parse_critique(completion.text)
        except ValueError:
            messages = [
                *messages,
                Message(role=Role.ASSISTANT, content=completion.text),
                Message(role=Role.USER, content=_REPAIR_INSTRUCTION),
            ]
            continue
        return CritiqueResult(
            critique=critique,
            usage=usage,
            prompt_id=prompt.id,
            fallback=False,
            raw=last_raw,
        )

    # All parse attempts failed; fail open to PASS to avoid wasting revisions
    return CritiqueResult(
        critique=Critique(verdict=CriticVerdict.PASS, issues=()),
        usage=usage,
        prompt_id=prompt.id,
        fallback=True,
        raw=last_raw,
    )
