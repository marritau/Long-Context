"""Personalized PageRank for evidence coverage."""

from __future__ import annotations

import math


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Stable softmax."""

    if not values:
        return []
    temperature = max(temperature, 1e-6)
    scaled = [value / temperature for value in values]
    high = max(scaled)
    exp_values = [math.exp(value - high) for value in scaled]
    total = sum(exp_values)
    if total == 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exp_values]


def normalize_seed(seed: list[float]) -> list[float]:
    """Normalize non-negative seed scores."""

    if not seed:
        return []
    clipped = [max(0.0, value) for value in seed]
    total = sum(clipped)
    if total == 0.0:
        return [1.0 / len(seed)] * len(seed)
    return [value / total for value in clipped]


def personalized_pagerank(
    graph: list[list[float]],
    seed: list[float],
    alpha: float = 0.85,
    iters: int = 50,
    tolerance: float = 1e-8,
) -> list[float]:
    """Run Personalized PageRank over a row-stochastic matrix."""

    n = len(graph)
    if n == 0:
        return []
    if len(seed) != n:
        raise ValueError("seed length must match graph size")

    restart = normalize_seed(seed)
    page = restart[:]
    alpha = min(max(alpha, 0.0), 1.0)

    for _ in range(max(1, iters)):
        incoming = [0.0] * n
        for i, row in enumerate(graph):
            for j, transition in enumerate(row):
                incoming[j] += page[i] * transition
        new_page = [(1.0 - alpha) * restart[i] + alpha * incoming[i] for i in range(n)]
        delta = sum(abs(new_page[i] - page[i]) for i in range(n))
        page = new_page
        if delta < tolerance:
            break
    return page
