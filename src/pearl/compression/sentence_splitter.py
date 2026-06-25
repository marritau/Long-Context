"""Sentence splitting utilities."""

from __future__ import annotations

import re

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""

    parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])
