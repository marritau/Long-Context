from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from offline_stage_c import fit_model, metrics, predict
from offline_v10_analysis import build_rows


TARGET_KEY = "candidate_log_odds"
TARGET_NAME = "candidate_competition_score"
BLOCK_PRIMITIVES = (
    "knorm_head_mean",
    "knorm_head_q90",
    "query_attention_head_mean",
    "query_attention_head_max",
    "query_attention_head_q90",
    "query_attention_head_std",
)
EDGE_PRIMITIVES = (
    "key_cohesion_head_mean",
    "key_cohesion_head_q90",
    "later_to_earlier_attention_head_mean",
    "later_to_earlier_attention_head_max",
    "later_to_earlier_attention_head_q90",
    "query_block_entropy_head_mean",
    "query_block_entropy_head_q90",
)


def _phase_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    early, middle, late = np.array_split(array, 3)
    return {
        "all": float(array.mean()),
        "early": float(early.mean()),
        "middle": float(middle.mean()),
        "late": float(late.mean()),
        "late_minus_early": float(late.mean() - early.mean()),
    }


def summarize_capture(payload: dict[str, Any], skip_first_layers: int = 2) -> dict[str, Any]:
    layers = sorted(payload["layers"], key=lambda row: int(row["layer_index"]))[skip_first_layers:]
    n_blocks = len(layers[0]["blocks"])
    n_edges = len(layers[0]["edges"])
    blocks: list[dict[str, float]] = []
    edges: list[dict[str, float]] = []
    for block in range(n_blocks):
        row: dict[str, float] = {}
        for primitive in BLOCK_PRIMITIVES:
            for phase, value in _phase_values(
                [float(layer["blocks"][block][primitive]) for layer in layers]
            ).items():
                row[f"{primitive}__{phase}"] = value
        blocks.append(row)
    for boundary in range(n_edges):
        row = {}
        for primitive in EDGE_PRIMITIVES:
            for phase, value in _phase_values(
                [float(layer["edges"][boundary][primitive]) for layer in layers]
            ).items():
                row[f"{primitive}__{phase}"] = value
        for name, value in payload["boundary_surprise"][boundary].items():
            row[f"boundary_surprise__{name}"] = float(value)
        edges.append(row)
    return {
        "blocks": blocks,
        "edges": edges,
        "evidence_order": payload.get("evidence_order"),
        "supporting_blocks": payload.get("supporting_blocks"),
    }


def legacy_unary_from_capture(payload: dict[str, Any], skip_first_layers: int = 2) -> list[float]:
    """Reconstruct the exact v10 candidate-proposal unary from v11 KNorm captures."""
    layers = sorted(payload["layers"], key=lambda row: int(row["layer_index"]))[skip_first_layers:]
    normalized = []
    for layer in layers:
        values = np.asarray(
            [float(block["knorm_head_mean"]) for block in layer["blocks"]], dtype=np.float64
        )
        scale = float(values.std())
        normalized.append(np.zeros_like(values) if scale < 1e-10 else (values - values.mean()) / scale)
    return np.mean(np.stack(normalized), axis=0).astype(float).tolist()


def summarize_capture_v2(
    payload: dict[str, Any],
    skip_first_layers: int = 2,
    legacy_layers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Length-transferable rich features with explicit query-conditioned pair terms."""
    summary = summarize_capture(payload, skip_first_layers)
    layers = sorted(payload["layers"], key=lambda row: int(row["layer_index"]))[skip_first_layers:]
    old_layers = (
        sorted(legacy_layers, key=lambda row: int(row["layer_index"]))[skip_first_layers:]
        if legacy_layers is not None
        else None
    )
    n_blocks = len(summary["blocks"])
    content_lengths = payload.get("layout", {}).get("block_content_lengths", [1] * n_blocks)
    data_start = int(payload.get("layout", {}).get("data_start", 0))
    block_size = int(payload.get("layout", {}).get("block_size", 1))
    for boundary in range(n_blocks - 1):
        edge = summary["edges"][boundary]
        query_pair_layers: dict[str, list[float]] = defaultdict(list)
        normalized_direction = []
        for layer in layers:
            masses = np.asarray(
                [float(block["query_attention_head_mean"]) for block in layer["blocks"]],
                dtype=np.float64,
            )
            probabilities = masses / max(float(masses.sum()), 1e-12)
            left, right = float(probabilities[boundary]), float(probabilities[boundary + 1])
            query_pair_layers["query_pair_geomean"].append(float(np.sqrt(max(0.0, left * right))))
            query_pair_layers["query_pair_min"].append(min(left, right))
            query_pair_layers["query_pair_sum"].append(left + right)
            query_pair_layers["query_pair_balance"].append(
                1.0 - abs(left - right) / max(left + right, 1e-12)
            )
            right_start = data_start + (boundary + 1) * block_size
            right_length = int(content_lengths[boundary + 1])
            left_length = int(content_lengths[boundary])
            expected = float(
                np.mean([left_length / max(1, right_start + offset + 1) for offset in range(right_length)])
            )
            observed = float(layer["edges"][boundary]["later_to_earlier_attention_head_mean"])
            normalized_direction.append(float(np.log((observed + 1e-8) / (expected + 1e-8))))
        for name, values in query_pair_layers.items():
            for phase, value in _phase_values(values).items():
                edge[f"{name}__{phase}"] = value
        for phase, value in _phase_values(normalized_direction).items():
            edge[f"direction_log_lift__{phase}"] = value
        source_layers = old_layers if old_layers is not None else layers
        if "hidden_cosine" in source_layers[0]["edges"][boundary]:
            for primitive in ("hidden_cosine", "hidden_jump"):
                values = [float(layer["edges"][boundary][primitive]) for layer in source_layers]
                for phase, value in _phase_values(values).items():
                    edge[f"{primitive}__{phase}"] = value
    summary["legacy_unary"] = legacy_unary_from_capture(payload, skip_first_layers)
    return summary


def load_rich_captures(root: Path, skip_first_layers: int = 2) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "rich_capture").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        output[str(payload["sample_id"])] = summarize_capture(payload, skip_first_layers)
    return output


def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {f"{prefix}_mean": float(np.mean(values)), f"{prefix}_max": float(np.max(values))}


def action_features(mask: list[int], capture: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    selected = set(mask)
    n_blocks = len(capture["blocks"])
    rich: dict[str, float] = {
        "geometry__position_mean": float(np.mean(mask) / (n_blocks - 1)),
        "geometry__position_std": float(np.std(mask) / (n_blocks - 1)),
        "geometry__keeps_first": float(0 in selected),
        "geometry__keeps_last": float(n_blocks - 1 in selected),
        "geometry__adjacency_fraction": float(
            sum(index + 1 in selected for index in selected) / max(1, len(mask) - 1)
        ),
    }
    decomposable: dict[str, float] = {}
    block_names = sorted(capture["blocks"][0])
    for name in block_names:
        values = [capture["blocks"][index][name] for index in mask]
        rich.update(_aggregate(values, f"selected_block__{name}"))
        decomposable[f"unary_sum__{name}"] = float(np.sum(values))
    kept_edges = [index for index in range(n_blocks - 1) if index in selected and index + 1 in selected]
    cut_edges = [index for index in range(n_blocks - 1) if (index in selected) != (index + 1 in selected)]
    edge_names = sorted(capture["edges"][0])
    for name in edge_names:
        rich.update(_aggregate([capture["edges"][index][name] for index in kept_edges], f"kept_edge__{name}"))
        rich.update(_aggregate([capture["edges"][index][name] for index in cut_edges], f"cut_edge__{name}"))
        decomposable[f"pair_sum__{name}"] = float(
            np.sum([capture["edges"][index][name] for index in kept_edges])
        )
    return rich, decomposable


def action_features_v2(mask: list[int], capture: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Action features whose scale is stable when context length or budget changes."""
    rich, _ = action_features(mask, capture)
    selected = set(mask)
    n_blocks = len(capture["blocks"])
    decomposable: dict[str, float] = {}
    for name in sorted(capture["blocks"][0]):
        decomposable[f"unary_mean__{name}"] = float(
            np.mean([capture["blocks"][index][name] for index in mask])
        )
    kept_edges = [index for index in range(n_blocks - 1) if index in selected and index + 1 in selected]
    denominator = max(1, len(mask) - 1)
    for name in sorted(capture["edges"][0]):
        decomposable[f"pair_budget_mean__{name}"] = float(
            np.sum([capture["edges"][index][name] for index in kept_edges]) / denominator
        )
    return rich, decomposable


def build_actions(boundary_rows: list[dict[str, Any]], captures: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        documents[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    rich_names: list[str] | None = None
    dp_names: list[str] | None = None
    for sample_id, rows in documents.items():
        aliases: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            aliases[tuple(row["proposal_mask"])].append(row)
        for mask_tuple, members in aliases.items():
            mask = list(mask_tuple)
            features, dp_features = action_features(mask, captures[sample_id])
            current_rich = sorted(features)
            current_dp = sorted(dp_features)
            rich_names = current_rich if rich_names is None else rich_names
            dp_names = current_dp if dp_names is None else dp_names
            if rich_names != current_rich or dp_names != current_dp:
                raise ValueError("Action feature schema changed between documents")
            support = set(members[0]["supporting_blocks"])
            output.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_group": members[0]["counterfactual_group"],
                    "split": members[0]["split"],
                    "evidence_order": members[0]["evidence_order"],
                    "mask": mask,
                    "features": features,
                    "dp_features": dp_features,
                    "target": float(members[0]["proposal_utilities"][TARGET_KEY]),
                    "retains_support": support.issubset(mask),
                    "same_as_support_proposal": any(member["is_support_boundary"] for member in members),
                }
            )
    if rich_names is None or dp_names is None:
        raise ValueError("No deployment actions were built")
    return output, rich_names, dp_names


def _subset(rows: list[dict[str, Any]], indices: np.ndarray) -> list[dict[str, Any]]:
    return [rows[int(index)] for index in indices]


def _group_splits(rows: list[dict[str, Any]], folds: int):
    groups = np.asarray([row["counterfactual_group"] for row in rows])
    indices = np.arange(len(rows))
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    return splitter.split(indices, groups=groups)


def _ranknet_cv_scores(rows: list[dict[str, Any]], names: list[str], c_value: float, folds: int) -> np.ndarray:
    scores = np.zeros(len(rows), dtype=np.float64)
    for train_index, test_index in _group_splits(rows, folds):
        scaler, model, kept = fit_model(_subset(rows, train_index), names, c_value)
        scores[test_index] = predict(_subset(rows, test_index), names, scaler, model, kept)
    return scores


def nested_ranknet(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, list[float]]:
    output = np.zeros(len(rows), dtype=np.float64)
    selected: list[float] = []
    c_grid = (0.001, 0.01, 0.1, 1.0, 10.0)
    for train_index, test_index in _group_splits(rows, 5):
        train = _subset(rows, train_index)
        candidates = []
        for c_value in c_grid:
            inner = _ranknet_cv_scores(train, names, c_value, 4)
            candidates.append((metrics(train, inner)["mean_action_utility_regret"], c_value))
        c_value = min(candidates)[1]
        scaler, model, kept = fit_model(train, names, c_value)
        output[test_index] = predict(_subset(rows, test_index), names, scaler, model, kept)
        selected.append(c_value)
    return output, selected


def _dp_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[row["dp_features"][name] for name in names] for row in rows], dtype=np.float64)


def _fit_dp(train: list[dict[str, Any]], names: list[str], alpha: float):
    matrix = _dp_matrix(train, names)
    variable = np.std(matrix, axis=0) > 1e-10
    scaler = StandardScaler().fit(matrix[:, variable])
    model = Ridge(alpha=alpha).fit(scaler.transform(matrix[:, variable]), [row["target"] for row in train])
    return scaler, model, variable


def _predict_dp(rows: list[dict[str, Any]], names: list[str], fitted) -> np.ndarray:
    scaler, model, variable = fitted
    return model.predict(scaler.transform(_dp_matrix(rows, names)[:, variable]))


def nested_dp(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, list[float]]:
    output = np.zeros(len(rows), dtype=np.float64)
    selected: list[float] = []
    alpha_grid = (0.1, 1.0, 10.0, 100.0, 1000.0)
    for train_index, test_index in _group_splits(rows, 5):
        train = _subset(rows, train_index)
        candidates = []
        for alpha in alpha_grid:
            inner = np.zeros(len(train), dtype=np.float64)
            for inner_train, inner_test in _group_splits(train, 4):
                fitted = _fit_dp(_subset(train, inner_train), names, alpha)
                inner[inner_test] = _predict_dp(_subset(train, inner_test), names, fitted)
            candidates.append((metrics(train, inner)["mean_action_utility_regret"], alpha))
        alpha = min(candidates)[1]
        output[test_index] = _predict_dp(_subset(rows, test_index), names, _fit_dp(train, names, alpha))
        selected.append(alpha)
    return output, selected


def freeze_ranknet(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    candidates = []
    for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
        scores = _ranknet_cv_scores(rows, names, c_value, 5)
        candidates.append((metrics(rows, scores)["mean_action_utility_regret"], c_value))
    c_value = min(candidates)[1]
    scaler, model, kept = fit_model(rows, names, c_value)
    return {
        "kind": "ranknet",
        "C": c_value,
        "feature_names": kept,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "intercept": 0.0,
    }


def freeze_dp(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        scores = np.zeros(len(rows), dtype=np.float64)
        for train_index, test_index in _group_splits(rows, 5):
            fitted = _fit_dp(_subset(rows, train_index), names, alpha)
            scores[test_index] = _predict_dp(_subset(rows, test_index), names, fitted)
        candidates.append((metrics(rows, scores)["mean_action_utility_regret"], alpha))
    alpha = min(candidates)[1]
    scaler, model, variable = _fit_dp(rows, names, alpha)
    return {
        "kind": "decomposable_ridge",
        "alpha": alpha,
        "feature_names": [name for name, keep in zip(names, variable) if keep],
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficients": model.coef_.astype(float).tolist(),
        "intercept": float(model.intercept_),
    }


def freeze_dp_at_alpha(rows: list[dict[str, Any]], names: list[str], alpha: float) -> dict[str, Any]:
    """Freeze a Ridge ablation at a preregistered alpha from its parent model."""
    scaler, model, variable = _fit_dp(rows, names, alpha)
    return {
        "kind": "decomposable_ridge",
        "alpha": float(alpha),
        "feature_names": [name for name, keep in zip(names, variable) if keep],
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficients": model.coef_.astype(float).tolist(),
        "intercept": float(model.intercept_),
    }


def baseline_scores(rows: list[dict[str, Any]], contains: str) -> np.ndarray:
    names = [name for name in rows[0]["features"] if contains in name and name.endswith("__all_mean")]
    names = [name for name in names if name.startswith("selected_block") and name.endswith("_mean")]
    if not names:
        # Feature names end in "__all_mean_mean": aggregation suffix follows the phase suffix.
        names = [name for name in rows[0]["features"] if contains in name and "__all_mean" in name and name.endswith("_mean")]
    return np.asarray([sum(row["features"][name] for name in names) for row in rows])


def metrics_by_order(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    output = {"all": metrics(rows, scores)}
    for order in sorted({row["evidence_order"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["evidence_order"] == order]
        output[order] = metrics([rows[index] for index in indices], scores[indices])
    return output


def document_outcomes(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, dict[str, Any]]:
    documents: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        documents[row["sample_id"]].append(index)
    output: dict[str, dict[str, Any]] = {}
    for sample_id, indices in documents.items():
        chosen = max(indices, key=lambda index: float(scores[index]))
        best = max(float(rows[index]["target"]) for index in indices)
        output[sample_id] = {
            "family": rows[chosen]["counterfactual_group"],
            "retention": float(rows[chosen]["retains_support"]),
            "regret": best - float(rows[chosen]["target"]),
            "chosen_mask": rows[chosen]["mask"],
        }
    return output


def paired_family_bootstrap(
    rows: list[dict[str, Any]], proposed: np.ndarray, baseline: np.ndarray, draws: int = 10000
) -> dict[str, Any]:
    left = document_outcomes(rows, proposed)
    right = document_outcomes(rows, baseline)
    by_family: dict[str, list[str]] = defaultdict(list)
    for sample_id, outcome in left.items():
        by_family[outcome["family"]].append(sample_id)
    families = sorted(by_family)
    retention = []
    regret_reduction = []
    rng = np.random.default_rng(20261012)
    for _ in range(draws):
        sampled = rng.choice(families, size=len(families), replace=True)
        sample_ids = [sample_id for family in sampled for sample_id in by_family[str(family)]]
        retention.append(float(np.mean([left[s]["retention"] - right[s]["retention"] for s in sample_ids])))
        regret_reduction.append(float(np.mean([right[s]["regret"] - left[s]["regret"] for s in sample_ids])))
    return {
        "retention_lift": float(np.mean([left[s]["retention"] - right[s]["retention"] for s in left])),
        "retention_lift_family_bootstrap_95": np.quantile(retention, [0.025, 0.975]).astype(float).tolist(),
        "regret_reduction": float(np.mean([right[s]["regret"] - left[s]["regret"] for s in left])),
        "regret_reduction_family_bootstrap_95": np.quantile(regret_reduction, [0.025, 0.975]).astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage E: grouped offline audit of v11 rich features")
    parser.add_argument("v10_run", type=Path)
    parser.add_argument("rich_run", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.v10_run / "offline_stage_e").resolve()
    output.mkdir(parents=True, exist_ok=True)

    boundary_rows, manifest = build_rows(args.v10_run.resolve(), args.data.resolve())
    captures = load_rich_captures(args.rich_run.resolve())
    actions, names, dp_names = build_actions(boundary_rows, captures)

    feature_sets = {
        "rich_primary": names,
        "no_position": [name for name in names if not name.startswith("geometry__")],
        "knorm_only": [name for name in names if "knorm_" in name],
        "query_attention_only": [name for name in names if "query_attention_" in name],
        "no_query": [name for name in names if "query_attention_" not in name],
        "no_surprise": [name for name in names if "boundary_surprise" not in name],
        "no_direction": [name for name in names if "later_to_earlier" not in name],
    }
    scores: dict[str, np.ndarray] = {}
    hyperparameters: dict[str, list[float]] = {}
    for name, selected_names in feature_sets.items():
        print(f"[stage-e] nested RankNet: {name} ({len(selected_names)} features)", flush=True)
        scores[name], hyperparameters[name] = nested_ranknet(actions, selected_names)
    dp_feature_sets = {
        "calibrated_quadratic_dp": dp_names,
        "dp_unary_only": [name for name in dp_names if name.startswith("unary_sum__")],
        "dp_pair_only": [name for name in dp_names if name.startswith("pair_sum__")],
        "dp_no_direction": [name for name in dp_names if "later_to_earlier" not in name],
        "dp_no_surprise": [name for name in dp_names if "boundary_surprise" not in name],
    }
    for name, selected_names in dp_feature_sets.items():
        print(f"[stage-e] nested decomposable Ridge: {name} ({len(selected_names)} features)", flush=True)
        scores[name], hyperparameters[name] = nested_dp(actions, selected_names)
    scores["shared_block_knorm"] = baseline_scores(actions, "knorm_head_mean")
    scores["shared_block_query_attention"] = baseline_scores(actions, "query_attention_head_mean")
    scores["action_oracle"] = np.asarray([row["target"] for row in actions])

    results = {name: metrics_by_order(actions, value) for name, value in scores.items()}
    primary = results["rich_primary"]
    no_position = results["no_position"]
    baseline = results["shared_block_knorm"]
    orders = sorted({row["evidence_order"] for row in actions})
    improves_orders = all(
        primary[order]["mean_action_utility_regret"] < baseline[order]["mean_action_utility_regret"]
        and primary[order]["action_support_retention"] >= baseline[order]["action_support_retention"]
        for order in orders
    )
    survives_position = (
        no_position["all"]["action_support_retention"] >= baseline["all"]["action_support_retention"]
        and no_position["all"]["mean_action_utility_regret"] < baseline["all"]["mean_action_utility_regret"]
    )
    decision = "PASS_TO_STAGE_F" if improves_orders and survives_position else "REFINE_BEFORE_STAGE_F"
    dp_candidate = (
        results["calibrated_quadratic_dp"]["all"]["mean_action_utility_regret"]
        <= primary["all"]["mean_action_utility_regret"]
        and results["calibrated_quadratic_dp"]["all"]["action_support_retention"]
        >= primary["all"]["action_support_retention"]
    )
    bootstrap = {
        "rich_primary_vs_shared_block_knorm": paired_family_bootstrap(
            actions, scores["rich_primary"], scores["shared_block_knorm"]
        ),
        "query_attention_only_vs_shared_block_knorm": paired_family_bootstrap(
            actions, scores["query_attention_only"], scores["shared_block_knorm"]
        ),
        "quadratic_dp_vs_shared_block_knorm": paired_family_bootstrap(
            actions, scores["calibrated_quadratic_dp"], scores["shared_block_knorm"]
        ),
        "quadratic_dp_vs_dp_unary_only": paired_family_bootstrap(
            actions, scores["calibrated_quadratic_dp"], scores["dp_unary_only"]
        ),
        "quadratic_dp_vs_query_attention_only": paired_family_bootstrap(
            actions, scores["calibrated_quadratic_dp"], scores["query_attention_only"]
        ),
    }
    interaction_increment_supported = (
        bootstrap["quadratic_dp_vs_dp_unary_only"]["regret_reduction_family_bootstrap_95"][0] > 0.0
        or bootstrap["quadratic_dp_vs_dp_unary_only"]["retention_lift_family_bootstrap_95"][0] > 0.0
    )
    frozen_models = {
        "query_attention_ranknet": freeze_ranknet(actions, feature_sets["query_attention_only"]),
        "quadratic_dp_no_direction": freeze_dp(actions, dp_feature_sets["dp_no_direction"]),
        "quadratic_dp_full": freeze_dp(actions, dp_feature_sets["calibrated_quadratic_dp"]),
    }
    report = {
        "stage": "E",
        "decision": decision,
        "dp_admitted_to_stage_f_as_candidate": dp_candidate,
        "interaction_increment_statistically_supported": interaction_increment_supported,
        "target": {
            "stored_key": TARGET_KEY,
            "scientific_name": TARGET_NAME,
            "note": "Length-normalized candidate score contrast; not calibrated probabilistic log-odds.",
        },
        "cohort": manifest["offline_cohort"],
        "captures_available": len(captures),
        "actions": len(actions),
        "families": len({row["counterfactual_group"] for row in actions}),
        "evidence_orders": Counter(row["evidence_order"] for row in actions),
        "feature_counts": {name: len(values) for name, values in feature_sets.items()} | {name: len(values) for name, values in dp_feature_sets.items()},
        "outer_cv": "5-fold GroupKFold by counterfactual family with inner 4-fold family-grouped tuning",
        "hyperparameters": hyperparameters,
        "results": results,
        "paired_family_bootstrap": bootstrap,
        "gate_checks": {"improves_shared_knorm_in_each_order": improves_orders, "survives_no_position": survives_position},
        "metadata_note": "Serialized layout is the original document layout (suffix_length=0); capture protocol and nonzero query features confirm the disposable query suffix was used internally.",
        "frozen_models": frozen_models,
    }
    (output / "stage_e.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage E: grouped offline policy audit",
        "",
        f"**Decision: {decision}**",
        f"**Quadratic DP admitted as a Stage F candidate: {dp_candidate}**",
        f"**Interaction increment statistically supported: {interaction_increment_supported}**",
        "",
        "| Method | Support retention | MRR | Utility regret |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        overall = result["all"]
        lines.append(
            f"| {name} | {overall['action_support_retention']:.1%} | {overall['support_action_mrr']:.3f} | {overall['mean_action_utility_regret']:.3f} |"
        )
    lines.extend(["", f"Families: {report['families']}; actions: {len(actions)}; captures: {len(captures)}.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    prediction_rows = []
    outcomes = {name: document_outcomes(actions, value) for name, value in scores.items()}
    for sample_id in sorted(outcomes["rich_primary"]):
        prediction_rows.append({"sample_id": sample_id, **{name: value[sample_id] for name, value in outcomes.items()}})
    (output / "oof_predictions.json").write_text(json.dumps(prediction_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "frozen_models.json").write_text(json.dumps(frozen_models, ensure_ascii=False, indent=2), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
