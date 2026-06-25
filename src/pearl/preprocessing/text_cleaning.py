"""Text cleaning helpers."""

from __future__ import annotations

import re

_SPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readable text."""

    return _SPACE_RE.sub(" ", text).strip()


def strip_control_chars(text: str) -> str:
    """Remove non-printable control characters."""

    return "".join(ch for ch in text if ch.isprintable() or ch.isspace())


def clean_text(text: str) -> str:
    """Default text cleaning pipeline."""

    return normalize_whitespace(strip_control_chars(text))
