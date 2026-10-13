from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from bikv.packing import CacheLayout


FEATURE_NAMES = [
    "left_unary",
    "right_unary",
    "min_unary",
    "max_unary",
    "unary_abs_diff",
    "hidden_cosine",
    "hidden_jump",
    "left_question_cosine",
    "right_question_cosine",
    "min_question_cosine",
    "max_question_cosine",
    "boundary_position",
]


@dataclass
class LayerFeatures:
    layer_index: int
    unary: list[float]
    edges: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {"layer_index": self.layer_index, "unary": self.unary, "edges": self.edges}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    value = F.cosine_similarity(left.float().unsqueeze(0), right.float().unsqueeze(0), dim=-1)
    return float(value.item())


def extract_layer_features(
    hidden_states: torch.Tensor,
    token_scores: torch.Tensor,
    layout: CacheLayout,
    layer_index: int,
    n_layers: int,
) -> LayerFeatures:
    if hidden_states.shape[0] != 1 or token_scores.shape[0] != 1:
        raise ValueError("The pilot currently supports batch size 1")
    if hidden_states.shape[1] != layout.original_length:
        raise ValueError(
            f"Expected {layout.original_length} hidden-state positions, got {hidden_states.shape[1]}"
        )

    scores = token_scores.float().mean(dim=1)[0]
    block_means = []
    unary = []
    for block_index in range(layout.n_blocks):
        start = layout.data_start + block_index * layout.block_size
        content_length = (
            int(layout.block_content_lengths[block_index])
            if layout.block_content_lengths
            else layout.block_size
        )
        if not 1 <= content_length <= layout.block_size:
            raise ValueError(f"Invalid content length {content_length} for block {block_index}")
        end = start + content_length
        block_means.append(hidden_states[0, start:end].float().mean(dim=0))
        unary.append(float(scores[start:end].mean().item()))

    suffix_start = layout.data_end
    if layout.suffix_length > 0:
        question_mean = hidden_states[0, suffix_start : suffix_start + layout.suffix_length].float().mean(dim=0)
    else:
        question_mean = hidden_states[0, : layout.prefix_length].float().mean(dim=0)

    question_cosines = [_cosine(block, question_mean) for block in block_means]
    edges = []
    for boundary in range(layout.n_blocks - 1):
        left = block_means[boundary]
        right = block_means[boundary + 1]
        left_unary = unary[boundary]
        right_unary = unary[boundary + 1]
        left_query = question_cosines[boundary]
        right_query = question_cosines[boundary + 1]
        row = {
            "left_unary": left_unary,
            "right_unary": right_unary,
            "min_unary": min(left_unary, right_unary),
            "max_unary": max(left_unary, right_unary),
            "unary_abs_diff": abs(left_unary - right_unary),
            "hidden_cosine": _cosine(left, right),
            "hidden_jump": float(torch.linalg.vector_norm(left - right).item() / max(1, left.numel()) ** 0.5),
            "left_question_cosine": left_query,
            "right_question_cosine": right_query,
            "min_question_cosine": min(left_query, right_query),
            "max_question_cosine": max(left_query, right_query),
            "boundary_position": float((boundary + 1) / layout.n_blocks),
        }
        edges.append(row)
    return LayerFeatures(layer_index=layer_index, unary=unary, edges=edges)


def feature_vector(row: dict[str, float]) -> list[float]:
    return [float(row[name]) for name in FEATURE_NAMES]


def summarize_layer_edge_features(
    layers: list[dict[str, Any]], skip_first_layers: int = 0
) -> tuple[list[str], list[dict[str, float]]]:
    """Preserve compact early/middle/late structure for one shared-mask ranker.

    This is additive to the legacy mean aggregation so old v10 models remain
    loadable. Query-related values come from the disposable diagnostic pass;
    its cache is not used by the scored compressed-document pass.
    """
    if not layers:
        raise ValueError("At least one captured layer is required")
    selected = sorted(layers, key=lambda row: int(row["layer_index"]))[skip_first_layers:]
    if len(selected) < 3:
        raise ValueError("Rich summaries require at least three retained layers")
    n_edges = len(selected[0]["edges"])
    if any(len(layer["edges"]) != n_edges for layer in selected):
        raise ValueError("Captured layers disagree on edge count")
    base_names = [name for name in FEATURE_NAMES if name != "boundary_position"]
    suffixes = ("mean", "std", "early", "middle", "late", "late_minus_early")
    names = [f"{name}__{suffix}" for name in base_names for suffix in suffixes]
    names.append("boundary_position")
    rows: list[dict[str, float]] = []
    for boundary in range(n_edges):
        row: dict[str, float] = {}
        for name in base_names:
            values = [float(layer["edges"][boundary][name]) for layer in selected]
            mean = sum(values) / len(values)
            quotient, remainder = divmod(len(values), 3)
            sizes = [quotient + int(index < remainder) for index in range(3)]
            segments = []
            start = 0
            for size in sizes:
                segments.append(values[start : start + size])
                start += size
            segment_means = [sum(segment) / len(segment) for segment in segments]
            row[f"{name}__mean"] = mean
            row[f"{name}__std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            row[f"{name}__early"] = segment_means[0]
            row[f"{name}__middle"] = segment_means[1]
            row[f"{name}__late"] = segment_means[2]
            row[f"{name}__late_minus_early"] = segment_means[2] - segment_means[0]
        row["boundary_position"] = float(selected[0]["edges"][boundary]["boundary_position"])
        rows.append(row)
    return names, rows


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float32)
    std = float(tensor.std(unbiased=False).item())
    mean = float(tensor.mean().item())
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(float(value) - mean) / std for value in values]
