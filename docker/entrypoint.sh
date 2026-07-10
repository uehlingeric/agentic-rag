#!/bin/sh
# First-boot bootstrap: `docker compose up` on a fresh volume must serve cited
# answers with no manual steps, so a missing index triggers ingest + index
# before serving. Subsequent boots skip straight to serve.
set -e

DATA_DIR="${AGENTIC_RAG_DATA_DIR:-data}"

if [ "$1" = "serve" ] && [ ! -f "$DATA_DIR/index/manifest.json" ]; then
    echo "no index at $DATA_DIR/index — running first-boot ingest + index" >&2
    agentic-rag ingest
    agentic-rag index
fi

exec agentic-rag "$@"
