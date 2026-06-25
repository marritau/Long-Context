"""Small graph centrality helpers."""

from __future__ import annotations


def weighted_degree(graph: list[list[float]]) -> list[float]:
    """Return weighted outgoing degree for each node."""

    return [sum(row) for row in graph]
