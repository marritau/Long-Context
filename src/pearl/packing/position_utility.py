"""Position utility curves."""

from __future__ import annotations

import math


def lost_middle_utility(position: float, rho: float = 0.35, kappa: float = 2.0) -> float:
    """U-shaped utility: beginning and end are preferred over the middle."""

    position = min(max(position, 0.0), 1.0)
    return 1.0 - rho * (math.sin(math.pi * position) ** kappa)


def normalized_position(index: int, n_items: int) -> float:
    """Map a list index to [0, 1]."""

    if n_items <= 1:
        return 0.0
    return index / (n_items - 1)
