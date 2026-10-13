from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence


@dataclass(frozen=True)
class SelectionResult:
    selected: list[int]
    objective: float
    used_budget: int


def select_chain_budget(
    unary: Sequence[float],
    edges: Sequence[float],
    costs: Sequence[int],
    budget: int,
    exact_budget: bool = False,
) -> SelectionResult:
    """Solve a chain-structured quadratic knapsack exactly."""
    n = len(unary)
    if len(costs) != n or len(edges) != max(0, n - 1):
        raise ValueError("Incompatible unary, edge and cost lengths")
    if budget < 0 or any(cost <= 0 for cost in costs):
        raise ValueError("Budget must be non-negative and costs must be positive")

    negative = float("-inf")
    dp = [[[negative, negative] for _ in range(budget + 1)] for _ in range(n + 1)]
    parent: list[list[list[tuple[int, int] | None]]] = [
        [[None, None] for _ in range(budget + 1)] for _ in range(n + 1)
    ]
    dp[0][0][0] = 0.0

    for index in range(n):
        for used in range(budget + 1):
            for previous in (0, 1):
                current = dp[index][used][previous]
                if current == negative:
                    continue
                if current > dp[index + 1][used][0]:
                    dp[index + 1][used][0] = current
                    parent[index + 1][used][0] = (previous, 0)

                new_used = used + int(costs[index])
                if new_used <= budget:
                    pair_bonus = float(edges[index - 1]) if index > 0 and previous else 0.0
                    candidate = current + float(unary[index]) + pair_bonus
                    if candidate > dp[index + 1][new_used][1]:
                        dp[index + 1][new_used][1] = candidate
                        parent[index + 1][new_used][1] = (previous, 1)

    best_value = negative
    best_used = 0
    best_previous = 0
    candidate_budgets = [budget] if exact_budget else range(budget + 1)
    for used in candidate_budgets:
        for previous in (0, 1):
            value = dp[n][used][previous]
            if value > best_value:
                best_value, best_used, best_previous = value, used, previous

    if best_value == negative:
        qualifier = "exactly" if exact_budget else "at most"
        raise ValueError(f"No selection uses {qualifier} budget={budget}")

    selected = []
    used = best_used
    previous = best_previous
    for index in range(n, 0, -1):
        state = parent[index][used][previous]
        if state is None:
            break
        old_previous, chosen = state
        if chosen:
            selected.append(index - 1)
            used -= int(costs[index - 1])
        previous = old_previous
    selected.reverse()
    return SelectionResult(selected=selected, objective=best_value, used_budget=best_used)


def brute_force_chain_budget(
    unary: Sequence[float],
    edges: Sequence[float],
    costs: Sequence[int],
    budget: int,
    exact_budget: bool = False,
) -> SelectionResult:
    best = SelectionResult([], float("-inf"), 0)
    for mask in product((0, 1), repeat=len(unary)):
        used = sum(cost for cost, chosen in zip(costs, mask) if chosen)
        if used > budget or (exact_budget and used != budget):
            continue
        value = sum(score for score, chosen in zip(unary, mask) if chosen)
        value += sum(edges[index] for index in range(len(edges)) if mask[index] and mask[index + 1])
        if value > best.objective:
            best = SelectionResult([index for index, chosen in enumerate(mask) if chosen], float(value), used)
    return best


def select_chain_with_cut_penalty(
    unary: Sequence[float],
    continuity: Sequence[float],
    costs: Sequence[int],
    budget: int,
    exact_budget: bool = False,
) -> SelectionResult:
    """Maximize unary utility while penalizing cuts through coherent boundaries.

    The objective is sum(u_i*x_i) - sum(c_i*|x_i-x_(i+1)|). Unlike a
    positive x_i*x_(i+1) bonus, it does not reward retaining an irrelevant
    adjacent pair; it only charges the selector for keeping half of a unit.
    """
    n = len(unary)
    if len(costs) != n or len(continuity) != max(0, n - 1):
        raise ValueError("Incompatible unary, continuity and cost lengths")
    if any(float(value) < 0 for value in continuity):
        raise ValueError("Continuity penalties must be non-negative")
    if n == 0:
        return SelectionResult([], 0.0, 0)

    negative = float("-inf")
    dp = [[[negative, negative] for _ in range(budget + 1)] for _ in range(n + 1)]
    parent: list[list[list[tuple[int, int] | None]]] = [
        [[None, None] for _ in range(budget + 1)] for _ in range(n + 1)
    ]
    dp[0][0][0] = 0.0
    dp[0][0][1] = 0.0

    for index in range(n):
        for used in range(budget + 1):
            for previous in (0, 1):
                current = dp[index][used][previous]
                if current == negative:
                    continue
                for chosen in (0, 1):
                    new_used = used + (int(costs[index]) if chosen else 0)
                    if new_used > budget:
                        continue
                    cut = float(continuity[index - 1]) if index > 0 and chosen != previous else 0.0
                    candidate = current + (float(unary[index]) if chosen else 0.0) - cut
                    if candidate > dp[index + 1][new_used][chosen]:
                        dp[index + 1][new_used][chosen] = candidate
                        parent[index + 1][new_used][chosen] = (previous, chosen)

    best_value = negative
    best_used = 0
    best_state = 0
    for used in ([budget] if exact_budget else range(budget + 1)):
        for state in (0, 1):
            if dp[n][used][state] > best_value:
                best_value, best_used, best_state = dp[n][used][state], used, state
    if best_value == negative:
        raise ValueError("No feasible selection for requested budget")

    selected: list[int] = []
    used = best_used
    state = best_state
    for index in range(n, 0, -1):
        previous = parent[index][used][state]
        if previous is None:
            break
        old_state, chosen = previous
        if chosen:
            selected.append(index - 1)
            used -= int(costs[index - 1])
        state = old_state
    selected.reverse()
    return SelectionResult(selected, best_value, best_used)
