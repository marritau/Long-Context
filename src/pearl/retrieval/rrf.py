"""Reciprocal Rank Fusion utilities."""

from __future__ import annotations


def ranks_from_scores(scores: list[float], descending: bool = True) -> dict[int, int]:
    """Convert scores to one-based ranks."""

    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=descending)
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fuse ranked id lists with Reciprocal Rank Fusion."""

    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def top_ids_from_scores(scores: list[float], top_k: int | None = None) -> list[int]:
    """Return ids sorted by descending score."""

    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return order if top_k is None else order[:top_k]
