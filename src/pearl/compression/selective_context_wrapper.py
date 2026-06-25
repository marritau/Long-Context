"""Wrapper placeholder for Selective Context baseline."""

from __future__ import annotations


class SelectiveContextUnavailable(RuntimeError):
    """Raised when Selective Context is requested but not installed."""


def compress_with_selective_context(*args, **kwargs):
    """Placeholder for future Selective Context integration."""

    raise SelectiveContextUnavailable(
        "Selective Context integration is optional. Install and wire the external package first."
    )
