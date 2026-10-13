from __future__ import annotations

from itertools import combinations
import math
import re
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
STOP = {
    "a", "an", "and", "at", "by", "every", "for", "from", "is", "it", "of", "only",
    "or", "record", "return", "the", "then", "through", "to", "uses", "with",
}


def content_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP}


def _zscore(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    scale = float(array.std())
    if scale < 1e-12:
        return [0.0] * len(values)
    return ((array - array.mean()) / scale).astype(float).tolist()


def relation_graph_scores(
    blocks: list[str], question: str, *, steps: int = 5, damping: float = 0.75
) -> list[float]:
    """Propagate query relevance through rare lexical links between blocks.

    This training-free control is deliberately independent of gold evidence and
    model captures. It tests whether explicit graph connectivity, rather than a
    second-order Ridge term alone, is required for distributed chains.
    """
    tokens = [content_tokens(block) for block in blocks]
    query = content_tokens(question)
    document_frequency: dict[str, int] = {}
    for row in tokens:
        for token in row:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    rare_limit = max(2, len(blocks) // 8)

    def weight(token: str) -> float:
        frequency = document_frequency.get(token, 0)
        if frequency > rare_limit:
            return 0.0
        return math.log((len(blocks) + 1) / (frequency + 1)) + 1.0

    seed = np.asarray(
        [sum(weight(token) for token in row & query) for row in tokens], dtype=np.float64
    )
    if float(seed.max()) <= 0.0:
        seed[:] = 1.0
    seed /= max(float(seed.max()), 1e-12)

    adjacency = np.zeros((len(blocks), len(blocks)), dtype=np.float64)
    for left, right in combinations(range(len(blocks)), 2):
        shared = tokens[left] & tokens[right]
        value = sum(weight(token) for token in shared)
        adjacency[left, right] = adjacency[right, left] = value
    row_sums = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(adjacency, row_sums, out=np.zeros_like(adjacency), where=row_sums > 0)
    scores = seed.copy()
    for _ in range(steps):
        scores = (1.0 - damping) * seed + damping * transition.T.dot(scores)
    return scores.astype(float).tolist()


def relation_graph_mask(blocks: list[str], question: str, budget: int) -> list[int]:
    scores = relation_graph_scores(blocks, question)
    return sorted(sorted(range(len(blocks)), key=lambda index: (-scores[index], index))[:budget])


def pair_primitives(
    capture: dict[str, Any], blocks: list[str], left: int, right: int
) -> dict[str, float]:
    if left == right:
        raise ValueError("A graph pair requires two distinct blocks")
    left, right = sorted((left, right))
    n_blocks = len(blocks)
    output: dict[str, float] = {}
    left_tokens, right_tokens = content_tokens(blocks[left]), content_tokens(blocks[right])
    shared = left_tokens & right_tokens
    union = left_tokens | right_tokens
    output["lexical_shared"] = float(len(shared))
    output["lexical_jaccard"] = len(shared) / max(1, len(union))
    output["lexical_containment"] = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
    output["distance_fraction"] = (right - left) / max(1, n_blocks - 1)
    output["is_adjacent"] = float(right == left + 1)

    left_row, right_row = capture["blocks"][left], capture["blocks"][right]
    for name in sorted(left_row):
        a, b = float(left_row[name]), float(right_row[name])
        output[f"min__{name}"] = min(a, b)
        output[f"max__{name}"] = max(a, b)
        output[f"absdiff__{name}"] = abs(a - b)
        output[f"product__{name}"] = a * b
        if "query_attention" in name:
            output[f"balance__{name}"] = 1.0 - abs(a - b) / max(abs(a) + abs(b), 1e-12)
            output[f"geomean__{name}"] = math.sqrt(max(0.0, a * b))
    return output


def graph_action_features(
    mask: list[int],
    capture: dict[str, Any],
    blocks: list[str],
    *,
    include_distance: bool = False,
    pair_permutation: list[int] | None = None,
) -> dict[str, float]:
    if len(mask) < 2:
        raise ValueError("Graph action features require at least two selected blocks")
    output: dict[str, float] = {}
    for name in sorted(capture["blocks"][0]):
        output[f"unary_mean__{name}"] = float(
            np.mean([float(capture["blocks"][index][name]) for index in mask])
        )
    if pair_permutation is None:
        pair_permutation = list(range(len(blocks)))
    if sorted(pair_permutation) != list(range(len(blocks))):
        raise ValueError("pair_permutation must contain every block index exactly once")
    rows = [
        pair_primitives(capture, blocks, pair_permutation[left], pair_permutation[right])
        for left, right in combinations(mask, 2)
    ]
    names = sorted(rows[0])
    if not include_distance:
        names = [name for name in names if name not in {"distance_fraction", "is_adjacent"}]
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        output[f"graph_pair_mean__{name}"] = float(values.mean())
        output[f"graph_pair_max__{name}"] = float(values.max())
    return output


def arbitrary_pair_candidates(
    capture: dict[str, Any],
    blocks: list[str],
    budget: int,
    shortlist_size: int = 12,
    question: str = "",
) -> list[list[int]]:
    n_blocks = len(blocks)
    if not 2 <= budget <= n_blocks:
        raise ValueError("Budget must be between two and the number of blocks")
    shortlist_size = min(n_blocks, max(budget, shortlist_size))
    query = [float(row.get("query_attention_head_mean__all", 0.0)) for row in capture["blocks"]]
    knorm = [float(value) for value in capture.get("legacy_unary", [0.0] * n_blocks)]
    relation = relation_graph_scores(blocks, question) if question else [0.0] * n_blocks
    composite = [
        left + right + graph
        for left, right, graph in zip(_zscore(query), _zscore(knorm), _zscore(relation))
    ]
    shortlist = sorted(range(n_blocks), key=lambda index: (-composite[index], index))[:shortlist_size]

    candidates: set[tuple[int, ...]] = set()
    unary_mask = tuple(sorted(shortlist[:budget]))
    candidates.add(unary_mask)
    pair_rows = {
        pair: pair_primitives(capture, blocks, *pair)
        for pair in combinations(sorted(shortlist), 2)
    }
    lexical_scale = max(
        1.0,
        max(row["lexical_shared"] for row in pair_rows.values()),
    )
    for anchor in pair_rows:
        selected = set(anchor)
        # Candidate A: arbitrary pair anchor plus the strongest unary blocks.
        for index in sorted(range(n_blocks), key=lambda item: (-composite[item], item)):
            if len(selected) >= budget:
                break
            selected.add(index)
        candidates.add(tuple(sorted(selected)))

        # Candidate B: same anchor followed by relation-aware graph expansion.
        selected = set(anchor)
        while len(selected) < budget:
            remaining = [index for index in range(n_blocks) if index not in selected]
            def expansion_score(index: int) -> tuple[float, float, int]:
                relation = 0.0
                for kept in selected:
                    pair = tuple(sorted((index, kept)))
                    row = pair_primitives(capture, blocks, *pair)
                    relation = max(
                        relation,
                        row["lexical_shared"] / lexical_scale + row["lexical_jaccard"],
                    )
                return relation, composite[index], -index
            selected.add(max(remaining, key=expansion_score))
        candidates.add(tuple(sorted(selected)))
    return [list(mask) for mask in sorted(candidates)]
