"""Budget allocation for query-aware compression."""

from __future__ import annotations

import math


def allocate_budgets(
    total_budget: int,
    importances: list[float],
    min_budget: int = 128,
    temperature: float = 1.0,
) -> list[int]:
    """Allocate integer token budgets under a total budget."""

    n = len(importances)
    if n == 0:
        return []
    if total_budget <= 0:
        return [0] * n

    min_budget = max(0, min_budget)
    if n * min_budget > total_budget:
        min_budget = max(1, total_budget // n)

    base = [min_budget] * n
    remaining = max(0, total_budget - sum(base))
    if remaining == 0:
        return base

    temperature = max(temperature, 1e-6)
    scaled = [value / temperature for value in importances]
    high = max(scaled)
    weights = [math.exp(value - high) for value in scaled]
    total_weight = sum(weights) or 1.0
    raw = [remaining * weight / total_weight for weight in weights]
    extra = [int(value) for value in raw]
    leftover = remaining - sum(extra)

    fractions = sorted(range(n), key=lambda idx: raw[idx] - extra[idx], reverse=True)
    for idx in fractions[:leftover]:
        extra[idx] += 1
    return [base[idx] + extra[idx] for idx in range(n)]
