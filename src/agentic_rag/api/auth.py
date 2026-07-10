"""Bearer token authentication for the API.

The expected token is bound at app creation (``create_app`` settings), not
read from global settings per request — the app instance and its token are
inseparable, which keeps injected test settings authoritative.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from hmac import compare_digest
from typing import Annotated

from fastapi import Header, HTTPException


def bearer_auth(expected_token: str) -> Callable[..., Awaitable[str]]:
    """Build a FastAPI dependency that requires ``Bearer <expected_token>``.

    Args:
        expected_token: The static API token requests must present.

    Returns:
        Async dependency returning the token, or raising 401 with a
        ``WWW-Authenticate: Bearer`` header on missing/malformed/wrong auth.
    """

    async def verify_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        def unauthorized(detail: str) -> HTTPException:
            return HTTPException(
                status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"}
            )

        if not authorization:
            raise unauthorized("Missing authorization header")

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise unauthorized("Invalid authorization header format")

        # compare_digest prevents timing attacks on the token comparison
        if not compare_digest(parts[1], expected_token):
            raise unauthorized("Invalid token")

        return parts[1]

    return verify_token
