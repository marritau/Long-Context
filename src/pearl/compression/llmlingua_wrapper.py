"""Wrapper placeholder for LLMLingua/LongLLMLingua baselines."""

from __future__ import annotations


class LLMLinguaUnavailable(RuntimeError):
    """Raised when LLMLingua is requested but not installed."""


def compress_with_llmlingua(*args, **kwargs):
    """Placeholder for future LLMLingua integration."""

    raise LLMLinguaUnavailable(
        "LLMLingua integration is optional. Install and wire the external package first."
    )
