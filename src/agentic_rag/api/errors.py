"""RFC 9457 problem+json error responses.

Refusals are NOT errors: a guardrail block or an out-of-corpus refusal is a
successful 200 answer whose ``refusal_reason`` names the machine-readable
cause. problem+json covers transport-level failures only — auth, validation,
rate limiting, provider outages, and unexpected server errors.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from http import HTTPStatus

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("agentic_rag.api")


def problem_response(
    status: int,
    title: str,
    detail: str,
    *,
    type_: str = "about:blank",
    headers: Mapping[str, str] | None = None,
    **extra: object,
) -> JSONResponse:
    """Build an RFC 9457 problem+json response.

    Args:
        status: HTTP status code.
        title: Short error summary (per RFC 9457).
        detail: Human-readable error details.
        type_: Problem type URI (default "about:blank").
        headers: Optional response headers (e.g. WWW-Authenticate).
        **extra: Additional fields to include in the response body.

    Returns:
        JSONResponse with application/problem+json media type.
    """
    body = {
        "type": type_,
        "title": title,
        "status": status,
        "detail": detail,
        **extra,
    }
    return JSONResponse(
        status_code=status,
        content=body,
        headers=headers,
        media_type="application/problem+json",
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert FastAPI HTTPException to problem+json, preserving headers."""
    title = HTTPStatus(exc.status_code).phrase
    return problem_response(exc.status_code, title, str(exc.detail), headers=exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to problem+json."""
    errors = [f"{'.'.join(str(x) for x in err['loc'][1:])} ({err['type']})" for err in exc.errors()]
    return problem_response(422, "Validation Error", f"Invalid request: {'; '.join(errors)}")


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Convert slowapi rate limit errors to problem+json."""
    return problem_response(429, "Too Many Requests", f"Rate limit exceeded: {exc.detail}")


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected exceptions (never leak details)."""
    logger.exception("Unhandled exception in request handler")
    return problem_response(
        500,
        "Internal Server Error",
        "An unexpected error occurred. Please try again later.",
    )
