"""Tiny file cache utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def cache_key(*parts: str) -> str:
    """Build a deterministic cache key."""

    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cache_path(root: str | Path, *parts: str, suffix: str = ".json") -> Path:
    """Return a cache file path."""

    return Path(root) / f"{cache_key(*parts)}{suffix}"
