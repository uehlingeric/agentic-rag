# syntax=docker/dockerfile:1

# Build stage: resolve the locked dependency set into a self-contained venv.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Runtime stage: slim Python, non-root, venv only (no uv, no build tools).
FROM python:3.14-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9

RUN useradd --create-home --uid 1000 app \
    && mkdir /data \
    && chown app:app /data

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config.yaml guardrails.yaml ./
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    AGENTIC_RAG_DATA_DIR=/data

USER app
VOLUME /data
EXPOSE 8000

# Liveness only: the API reports index readiness in the /health body.
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

ENTRYPOINT ["entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
