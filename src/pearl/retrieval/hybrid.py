"""Hybrid retrieval scoring helpers."""

from __future__ import annotations

from pearl.retrieval.rrf import reciprocal_rank_fusion, top_ids_from_scores


def minmax_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1]."""

    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-12:
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def combine_scores(score_sets: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    """Weighted combination of min-max normalized score arrays."""

    if not score_sets:
        return []
    n = len(next(iter(score_sets.values())))
    combined = [0.0] * n
    total_weight = sum(max(0.0, weights.get(name, 0.0)) for name in score_sets)
    if total_weight == 0.0:
        total_weight = 1.0
    for name, scores in score_sets.items():
        if len(scores) != n:
            raise ValueError("all score arrays must have the same length")
        weight = max(0.0, weights.get(name, 0.0)) / total_weight
        normalized = minmax_normalize(scores)
        for idx, score in enumerate(normalized):
            combined[idx] += weight * score
    return combined


def hybrid_rrf_ids(
    bm25_scores: list[float],
    dense_scores: list[float],
    top_m: int,
    rrf_k: int = 60,
) -> list[int]:
    """Return candidate ids from BM25+dense RRF."""

    bm25_order = top_ids_from_scores(bm25_scores)
    dense_order = top_ids_from_scores(dense_scores)
    fused = reciprocal_rank_fusion([bm25_order, dense_order], k=rrf_k)
    return sorted(fused, key=fused.get, reverse=True)[:top_m]
