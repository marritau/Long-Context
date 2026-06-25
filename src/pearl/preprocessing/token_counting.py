"""Lightweight token counting utilities.

The MVP uses a deterministic regex tokenizer so tests do not depend on model
tokenizers. Experiments can later swap this for `tiktoken` or a HF tokenizer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def iter_token_spans(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield `(token, char_start, char_end)` triples."""

    for match in TOKEN_RE.finditer(text):
        yield match.group(0), match.start(), match.end()


def simple_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into words and punctuation."""

    tokens = [match.group(0) for match in TOKEN_RE.finditer(text)]
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def count_tokens(text: str) -> int:
    """Count regex tokens in text."""

    return sum(1 for _ in TOKEN_RE.finditer(text))


def truncate_tokens(text: str, max_tokens: int, from_end: bool = False) -> str:
    """Return text truncated to at most `max_tokens` regex tokens."""

    if max_tokens <= 0:
        return ""
    spans = list(iter_token_spans(text))
    if len(spans) <= max_tokens:
        return text
    selected = spans[-max_tokens:] if from_end else spans[:max_tokens]
    return text[selected[0][1] : selected[-1][2]]
