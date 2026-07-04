"""Provider-neutral token counting.

Chunk sizing and context budgets need one consistent measure across providers;
tiktoken's o200k_base is used everywhere as the approximation. Vendor-exact
counts (billing) come from provider responses, not from this module.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def encode(text: str) -> list[int]:
    return _encoding().encode(text)


def decode(tokens: list[int]) -> str:
    return _encoding().decode(tokens)
