"""Selection strategies for evidence chunks."""

from __future__ import annotations

from pearl.data.schema import Chunk
from pearl.retrieval.dense import cosine


def mmr_select(
    chunks: list[Chunk],
    relevance: list[float],
    graph_scores: list[float] | None,
    budget: int,
    vectors: list[list[float]] | None = None,
    redundancy_penalty: float = 0.25,
    relevance_weight: float = 0.55,
    graph_weight: float = 0.45,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Select chunks under a token budget using MMR-style redundancy control."""

    if len(chunks) != len(relevance):
        raise ValueError("relevance length must match chunks length")
    if graph_scores is None:
        graph_scores = [0.0] * len(chunks)
    if len(graph_scores) != len(chunks):
        raise ValueError("graph_scores length must match chunks length")

    selected: list[int] = []
    remaining = set(range(len(chunks)))
    used_budget = 0
    while remaining:
        if max_chunks is not None and len(selected) >= max_chunks:
            break
        feasible = [idx for idx in remaining if chunks[idx].token_len + used_budget <= budget]
        if not feasible:
            break
        best_idx = max(
            feasible,
            key=lambda idx: _mmr_score(
                idx,
                selected,
                relevance,
                graph_scores,
                vectors,
                redundancy_penalty,
                relevance_weight,
                graph_weight,
            ),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
        used_budget += chunks[best_idx].token_len
    return [chunks[idx] for idx in selected]


def _mmr_score(
    idx: int,
    selected: list[int],
    relevance: list[float],
    graph_scores: list[float],
    vectors: list[list[float]] | None,
    redundancy_penalty: float,
    relevance_weight: float,
    graph_weight: float,
) -> float:
    redundancy = 0.0
    if vectors is not None and selected:
        redundancy = max(max(0.0, cosine(vectors[idx], vectors[j])) for j in selected)
    return (
        relevance_weight * relevance[idx]
        + graph_weight * graph_scores[idx]
        - redundancy_penalty * redundancy
    )
