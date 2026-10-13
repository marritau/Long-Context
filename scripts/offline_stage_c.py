from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from bikv.features import FEATURE_NAMES, summarize_layer_edge_features
from bikv.predictor import LinearInteractionModel
from offline_v10_analysis import build_rows, read_json


PRIMARY_UTILITY = "candidate_log_odds"


def zscore_values(values: list[float]) -> list[float]:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1e-8:
        return [0.0 for _ in values]
    return [(float(value) - mean) / scale for value in values]


def load_captures(run_root: Path, skip_first_layers: int) -> tuple[list[str], dict[tuple[str, int], dict[str, float]], dict[str, list[float]]]:
    feature_names: list[str] | None = None
    edges: dict[tuple[str, int], dict[str, float]] = {}
    unary: dict[str, list[float]] = {}
    for path in sorted((run_root / "capture").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        names, rows = summarize_layer_edge_features(payload["layers"], skip_first_layers)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Rich feature schemas differ between captures")
        sample_id = str(payload["sample_id"])
        for boundary, row in enumerate(rows):
            edges[(sample_id, boundary)] = row
        retained_layers = sorted(payload["layers"], key=lambda item: int(item["layer_index"]))[skip_first_layers:]
        normalized = [zscore_values([float(value) for value in layer["unary"]]) for layer in retained_layers]
        unary[sample_id] = [float(np.mean([layer[index] for layer in normalized])) for index in range(len(normalized[0]))]
    if feature_names is None:
        raise ValueError("No compressed capture files were found")
    return feature_names, edges, unary


def mask_features(mask: list[int], unary: list[float], n_blocks: int) -> dict[str, float]:
    selected = np.asarray([unary[index] for index in mask], dtype=np.float64)
    positions = np.asarray(mask, dtype=np.float64) / max(1, n_blocks - 1)
    adjacency = sum(right == left + 1 for left, right in zip(sorted(mask), sorted(mask)[1:]))
    return {
        "mask_unary_sum": float(selected.sum()),
        "mask_unary_mean": float(selected.mean()),
        "mask_unary_min": float(selected.min()),
        "mask_unary_max": float(selected.max()),
        "mask_position_mean": float(positions.mean()),
        "mask_position_std": float(positions.std()),
        "mask_adjacency_fraction": float(adjacency / max(1, len(mask) - 1)),
        "mask_keeps_first": float(0 in mask),
        "mask_keeps_last": float(n_blocks - 1 in mask),
    }


def build_action_rows(
    rows: list[dict[str, Any]], rich_edges: dict[tuple[str, int], dict[str, float]], unary: dict[str, list[float]]
) -> tuple[list[dict[str, Any]], list[str]]:
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        enriched = dict(row)
        enriched["rich_features"] = rich_edges[(row["sample_id"], row["boundary"])]
        documents[row["sample_id"]].append(enriched)
    output: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for sample_id, document in documents.items():
        aliases: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in document:
            aliases[tuple(row["proposal_mask"])].append(row)
        for mask, members in aliases.items():
            features: dict[str, float] = {}
            edge_names = sorted(members[0]["rich_features"])
            for name in edge_names:
                values = [member["rich_features"][name] for member in members]
                features[f"edge_mean__{name}"] = float(np.mean(values))
                features[f"edge_max__{name}"] = float(np.max(values))
            features.update(mask_features(list(mask), unary[sample_id], len(unary[sample_id])))
            current_names = sorted(features)
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError("Action feature schemas differ")
            support = set(members[0]["supporting_blocks"])
            output.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_group": members[0]["counterfactual_group"],
                    "split": members[0]["split"],
                    "mask": list(mask),
                    "features": features,
                    "target": float(members[0]["proposal_utilities"][PRIMARY_UTILITY]),
                    "retains_support": support.issubset(mask),
                    "same_as_support_proposal": any(member["is_support_boundary"] for member in members),
                    "singleton_retains_support": support.issubset(set(members[0]["singleton_mask"])),
                    "v10_score": float("-inf"),
                    "stage_b_score": float("-inf"),
                    "member_boundaries": [member["boundary"] for member in members],
                }
            )
    assert feature_names is not None
    return output, feature_names


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[row["features"][name] for name in names] for row in rows], dtype=np.float64)


def pairwise_arrays(rows: list[dict[str, Any]], transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    documents: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        documents[row["sample_id"]].append(index)
    x_out: list[np.ndarray] = []
    y_out: list[int] = []
    weights: list[float] = []
    for indices in documents.values():
        values = np.asarray([rows[index]["target"] for index in indices])
        gaps = [abs(float(values[i] - values[j])) for i in range(len(values)) for j in range(i + 1, len(values))]
        scale = max(float(np.median(gaps)), 1e-6)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                gap = float(values[left] - values[right])
                if abs(gap) <= 0.10 * scale:
                    continue
                difference = transformed[indices[left]] - transformed[indices[right]]
                label = int(gap > 0.0)
                weight = min(abs(gap) / scale, 4.0)
                x_out.extend((difference, -difference))
                y_out.extend((label, 1 - label))
                weights.extend((weight, weight))
    return np.asarray(x_out), np.asarray(y_out), np.asarray(weights)


def fit_model(rows: list[dict[str, Any]], names: list[str], c_value: float) -> tuple[StandardScaler, LogisticRegression, list[str]]:
    matrix = feature_matrix(rows, names)
    variable = np.std(matrix, axis=0) > 1e-10
    kept = [name for name, keep in zip(names, variable) if keep]
    scaler = StandardScaler().fit(matrix[:, variable])
    transformed = scaler.transform(matrix[:, variable])
    pair_x, pair_y, weights = pairwise_arrays(rows, transformed)
    model = LogisticRegression(C=c_value, fit_intercept=False, max_iter=5000)
    model.fit(pair_x, pair_y, sample_weight=weights)
    return scaler, model, kept


def predict(rows: list[dict[str, Any]], names: list[str], scaler: StandardScaler, model: LogisticRegression, kept: list[str]) -> np.ndarray:
    indices = [names.index(name) for name in kept]
    return model.decision_function(scaler.transform(feature_matrix(rows, names)[:, indices]))


def metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    documents: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        documents[row["sample_id"]].append(index)
    retention: list[float] = []
    exact: list[float] = []
    regret: list[float] = []
    reciprocal: list[float] = []
    for indices in documents.values():
        ranked = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
        chosen = rows[ranked[0]]
        retention.append(float(chosen["retains_support"]))
        exact.append(float(chosen["same_as_support_proposal"]))
        best = max(rows[index]["target"] for index in indices)
        regret.append(best - chosen["target"])
        support_rank = next((rank for rank, index in enumerate(ranked, 1) if rows[index]["retains_support"]), None)
        reciprocal.append(0.0 if support_rank is None else 1.0 / support_rank)
    return {
        "documents": float(len(documents)),
        "action_support_retention": float(np.mean(retention)),
        "support_action_mrr": float(np.mean(reciprocal)),
        "support_proposal_equivalence_at_1": float(np.mean(exact)),
        "mean_action_utility_regret": float(np.mean(regret)),
    }


def grouped_predictions(rows: list[dict[str, Any]], names: list[str], c_value: float, splits: int) -> np.ndarray:
    groups = np.asarray([row["counterfactual_group"] for row in rows])
    output = np.zeros(len(rows), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(splits, len(np.unique(groups))))
    all_indices = np.arange(len(rows))
    for train_index, test_index in splitter.split(all_indices, groups=groups):
        train = [rows[index] for index in train_index]
        test = [rows[index] for index in test_index]
        scaler, model, kept = fit_model(train, names, c_value)
        output[test_index] = predict(test, names, scaler, model, kept)
    return output


def choose_c(rows: list[dict[str, Any]], names: list[str]) -> tuple[float, list[dict[str, float]]]:
    grid = []
    for c_value in np.logspace(-4, 2, 13):
        scores = grouped_predictions(rows, names, float(c_value), 5)
        result = metrics(rows, scores)
        grid.append({"C": float(c_value), **result})
    best = min(grid, key=lambda item: (item["mean_action_utility_regret"], -item["support_action_mrr"], item["C"]))
    return float(best["C"]), grid


def train_and_evaluate(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], names: list[str]
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, float]]]:
    c_value, grid = choose_c(train, names)
    scaler, model, kept = fit_model(train, names, c_value)
    scores = predict(validation, names, scaler, model, kept)
    payload = {
        "C": c_value,
        "feature_names": kept,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "intercept": 0.0,
    }
    return metrics(validation, scores), payload, grid


def attach_baselines(action_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], run_root: Path) -> None:
    old_payload = read_json(run_root / "models" / "pair_model.json")
    old = LinearInteractionModel(**{key: old_payload[key] for key in (
        "feature_names", "mean", "scale", "coefficients", "intercept", "positive_threshold", "validation_metrics"
    )})
    stage_b = read_json(run_root / "offline_v11" / "stage_b.json")["model"]
    by_action = {(row["sample_id"], tuple(row["mask"])): row for row in action_rows}
    for row in boundary_rows:
        key = (row["sample_id"], tuple(row["proposal_mask"]))
        action = by_action[key]
        action["v10_score"] = max(action["v10_score"], old.predict(row["features"]))
        vector = np.asarray([row["features"][name] for name in stage_b["feature_names"]])
        standardized = (vector - np.asarray(stage_b["mean"])) / np.asarray(stage_b["scale"])
        score = float(np.dot(standardized, np.asarray(stage_b["coefficients"])))
        action["stage_b_score"] = max(action["stage_b_score"], score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C: rich layer-wise, action-aligned offline ranking")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    data_path = (args.data or run_root.parents[1] / "data" / "qualification.jsonl").resolve()
    output = (args.output or run_root / "offline_stage_c").resolve()
    output.mkdir(parents=True, exist_ok=True)

    boundary_rows, manifest = build_rows(run_root, data_path)
    _, rich_edges, unary = load_captures(run_root, int(manifest["config"]["feature_skip_first_layers"]))
    action_rows, names = build_action_rows(boundary_rows, rich_edges, unary)
    attach_baselines(action_rows, boundary_rows, run_root)
    train = [row for row in action_rows if row["split"] == "train"]
    validation = [row for row in action_rows if row["split"] == "validation"]
    proposed_metrics, primary_model, grid = train_and_evaluate(train, validation, names)
    c_value = float(primary_model["C"])
    kept = list(primary_model["feature_names"])
    all_oof = grouped_predictions(action_rows, names, c_value, len({row["counterfactual_group"] for row in action_rows}))
    no_position_names = [name for name in names if "position" not in name]
    no_query_names = [name for name in names if "question_cosine" not in name]
    mean_only_names = [
        name for name in names
        if name.startswith("mask_")
        or name.endswith("__mean")
        or name.endswith("__boundary_position")
    ]
    no_position_metrics, no_position_model, no_position_grid = train_and_evaluate(
        train, validation, no_position_names
    )
    no_query_metrics, no_query_model, no_query_grid = train_and_evaluate(
        train, validation, no_query_names
    )
    mean_only_metrics, mean_only_model, mean_only_grid = train_and_evaluate(
        train, validation, mean_only_names
    )
    methods = {
        "v10_ridge_boundary_then_action": metrics(validation, np.asarray([row["v10_score"] for row in validation])),
        "stage_b_ranknet_boundary_then_action": metrics(validation, np.asarray([row["stage_b_score"] for row in validation])),
        "stage_c_mean_only_action_ranknet": mean_only_metrics,
        "stage_c_rich_no_query_ranknet": no_query_metrics,
        "stage_c_rich_no_position_ranknet": no_position_metrics,
        "stage_c_rich_action_ranknet": proposed_metrics,
        "offline_action_oracle": metrics(validation, np.asarray([row["target"] for row in validation])),
    }
    proposed = methods["stage_c_rich_action_ranknet"]
    baseline = methods["v10_ridge_boundary_then_action"]
    decision = "PASS_TO_SHORT_GPU_PILOT" if (
        proposed["action_support_retention"] > baseline["action_support_retention"]
        and proposed["mean_action_utility_regret"] < baseline["mean_action_utility_regret"]
        and no_position_metrics["action_support_retention"] >= baseline["action_support_retention"]
    ) else "REFINE_REPRESENTATION"
    report = {
        "stage": "C",
        "decision": decision,
        "scientific_alignment": {
            "causal_protocol_changed": False,
            "factorial_estimand_changed": False,
            "final_training_unit": "unique exact-budget deployment mask",
            "final_target": "candidate_log_odds of the scored deployment mask",
            "interaction_role": "mechanism diagnostic, not substituted for final action utility",
            "sealed_test_used": False,
        },
        "cohort": manifest["offline_cohort"],
        "actions": {
            "train": len(train),
            "validation": len(validation),
            "mean_actions_per_document": len(action_rows) / len({row["sample_id"] for row in action_rows}),
        },
        "features": {"total": len(names), "retained_after_train_variance_filter": len(kept), "names": kept},
        "selected_C": c_value,
        "train_group_cv": {
            "primary_rich": grid,
            "no_position_ablation": no_position_grid,
            "no_query_ablation": no_query_grid,
            "mean_only_ablation": mean_only_grid,
        },
        "validation": methods,
        "development_leave_one_family_out": metrics(action_rows, all_oof),
        "models": {
            "primary_rich": primary_model,
            "no_position_ablation": no_position_model,
            "no_query_ablation": no_query_model,
            "mean_only_ablation": mean_only_model,
        },
    }
    (output / "stage_c.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage C: layer-wise action-aligned ranking", "", f"**Decision: {decision}**", "",
        "The clean cache protocol and matched-factorial estimand are unchanged. The final learner ranks unique exact-budget deployment masks directly.", "",
        "| Method | Support retention | Support action MRR | Utility regret |", "|---|---:|---:|---:|",
    ]
    for method, result in methods.items():
        lines.append(f"| {method} | {result['action_support_retention']:.1%} | {result['support_action_mrr']:.3f} | {result['mean_action_utility_regret']:.3f} |")
    lines.extend(["", f"Unique actions: {len(train)} train and {len(validation)} validation. Rich features retained: {len(kept)}/{len(names)}.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
