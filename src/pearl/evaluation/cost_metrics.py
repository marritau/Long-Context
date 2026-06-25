"""Cost-aware metrics."""

from __future__ import annotations

import math


def cost_normalized_score(answer_score: float, input_tokens: int) -> float:
    """Normalize quality by logarithmic input size."""

    return answer_score / math.log1p(max(1, input_tokens))


def linear_cost_score(answer_score: float, input_tokens: int, penalty_per_1k: float = 0.01) -> float:
    """Subtract a linear token-cost penalty."""

    return answer_score - penalty_per_1k * input_tokens / 1000.0
