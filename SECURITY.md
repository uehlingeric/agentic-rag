# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/uehlingeric/agentic-rag/security/advisories/new)
— do not open a public issue for anything exploitable. You should get a response
within a few days.

## Scope notes

- The guardrail layer's known limitations are documented deliberately in
  [docs/limitations.md](docs/limitations.md) and the red-team results in
  [docs/guardrails.md](docs/guardrails.md) (e.g., multilingual and homoglyph
  injection variants are annotated known misses). A new bypass **class**, or a
  bypass of a documented catch, is in scope — a variant of an annotated known
  miss is not.
- The API's static bearer token and per-token rate limits are a development
  posture, not a production auth story (see docs/limitations.md). Reports that
  amount to "static tokens are weak" are out of scope; flaws in how the token is
  checked (timing, bypass, header parsing) are in scope.
- Secrets never belong in the repo: CI runs gitleaks over full history. If you
  find a real credential anywhere in the history, report it privately.
