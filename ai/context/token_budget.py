"""Token budget estimation utilities."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~1.3 tokens/word) without a model tokenizer dependency."""
    if not text:
        return 0
    words = len(_WORD_RE.findall(text))
    return max(1, int(words * 1.3))


class TokenBudget:
    def __init__(self, total: int) -> None:
        self.total = total
        self.used = 0

    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def fits(self, text: str) -> bool:
        return estimate_tokens(text) <= self.remaining()

    def consume(self, text: str) -> bool:
        cost = estimate_tokens(text)
        if cost > self.remaining():
            return False
        self.used += cost
        return True
