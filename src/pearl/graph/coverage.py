"""Coverage helpers for graph-based evidence selection."""

from __future__ import annotations


def all_hop_covered(hop_hits: list[bool]) -> bool:
    """Return whether all required hops are covered."""

    return bool(hop_hits) and all(hop_hits)


def partial_hop_recall(hop_hits: list[bool]) -> float:
    """Fraction of covered hops."""

    if not hop_hits:
        return 0.0
    return sum(1 for hit in hop_hits if hit) / len(hop_hits)
