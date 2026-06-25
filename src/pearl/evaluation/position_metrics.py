"""Position-sensitivity metrics."""

from __future__ import annotations


def position_sensitivity_index(scores_by_position: dict[float, float]) -> float:
    """PSI = max score - min score."""

    if not scores_by_position:
        return 0.0
    values = list(scores_by_position.values())
    return max(values) - min(values)


def middle_drop(begin_score: float, middle_score: float, end_score: float) -> float:
    """Middle degradation compared with beginning/end average."""

    return (begin_score + end_score) / 2.0 - middle_score
