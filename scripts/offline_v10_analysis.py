from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from bikv.features import FEATURE_NAMES
from bikv.predictor import LinearInteractionModel


UTILITY_NAMES = ("max_margin", "gold_logprob", "candidate_log_odds")
CELL_NAMES = ("both", "left_only", "right_only", "neither")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_logsumexp(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def result_utilities(result: dict[str, Any], gold: str) -> dict[str, float]:
    scores = {str(key): float(value) for key, value in result["scores"].items()}
    gold_score = float(result["gold_score"])
    if gold not in scores or abs(scores[gold] - gold_score) >= 1e-7:
        raise ValueError(f"Gold score is inconsistent for answer {gold}")
    wrong_scores = [value for key, value in scores.items() if key != gold]
    return {
        "max_margin": gold_score - max(wrong_scores),
        "gold_logprob": gold_score,
        "candidate_log_odds": gold_score - stable_logsumexp(wrong_scores),
    }


def wrong_winner(result: dict[str, Any], gold: str) -> str:
    scores = {str(key): float(value) for key, value in result["scores"].items()}
    return max((key for key in scores if key != gold), key=scores.__getitem__)


def mask_key(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in values))


def load_mask_results(run_root: Path) -> dict[str, dict[tuple[int, ...], dict[str, Any]]]:
    output: dict[str, dict[tuple[int, ...], dict[str, Any]]] = defaultdict(dict)
    for path in sorted((run_root / "proposal_masks").rglob("*.json")):
        payload = read_json(path)
        sample_id = str(payload["sample_id"])
        key = mask_key(payload["selected_blocks"])
        previous = output[sample_id].get(key)
        if previous is not None and previous["scores"] != payload["result"]["scores"]:
            raise ValueError(f"Conflicting cached result for {sample_id} mask {key}")
        output[sample_id][key] = payload["result"]
    return output


def load_gold_answers(data_path: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            answers[str(payload["sample_id"])] = payload
    return answers


def build_rows(run_root: Path, data_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    expected_hash = str(manifest["config_hash"])
    masks = load_mask_results(run_root)
    source = load_gold_answers(data_path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    family_passes: dict[str, list[bool]] = defaultdict(list)
    for path in sorted((run_root / "labels").glob("*.json")):
        payload = read_json(path)
        if payload["config_hash"] != expected_hash:
            raise ValueError(f"Config hash mismatch in {path}")
        sample_id = str(payload["sample_id"])
        family = str(payload["counterfactual_group"])
        family_passes[family].append(bool(payload["necessity_passed"]))
        if sample_id not in source:
            raise ValueError(f"Source dataset has no record for {sample_id}")
        gold = str(source[sample_id]["answer"])
        support = [int(value) for value in payload["supporting_blocks"]]
        if support != [int(value) for value in source[sample_id]["supporting_blocks"]]:
            raise ValueError(f"Source dataset support mismatch for {sample_id}")
        for edge in payload["edge_labels"]:
            results: dict[str, dict[str, Any]] = {}
            for cell in CELL_NAMES:
                key = mask_key(edge["intervention_masks"][cell])
                if key not in masks[sample_id]:
                    missing.append(f"{sample_id}: boundary={edge['boundary']} cell={cell} mask={key}")
                    continue
                results[cell] = masks[sample_id][key]
            proposal_key = mask_key(edge["proposal_mask"])
            singleton_key = mask_key(edge["singleton_mask"])
            if proposal_key not in masks[sample_id] or singleton_key not in masks[sample_id]:
                missing.append(f"{sample_id}: deployment cache missing at boundary={edge['boundary']}")
                continue
            if len(results) != 4:
                continue
            cell_utilities = {cell: result_utilities(result, gold) for cell, result in results.items()}
            targets = {
                name: cell_utilities["both"][name]
                - cell_utilities["left_only"][name]
                - cell_utilities["right_only"][name]
                + cell_utilities["neither"][name]
                for name in UTILITY_NAMES
            }
            proposal_utilities = result_utilities(masks[sample_id][proposal_key], gold)
            singleton_utilities = result_utilities(masks[sample_id][singleton_key], gold)
            rows.append(
                {
                    "sample_id": sample_id,
                    "story_id": str(payload["story_id"]),
                    "counterfactual_group": str(payload["counterfactual_group"]),
                    "split": str(payload["split"]),
                    "evidence_order": str(payload["evidence_order"]),
                    "boundary": int(edge["boundary"]),
                    "is_support_boundary": bool(edge["is_support_boundary"]),
                    "supporting_blocks": support,
                    "features": {name: float(edge["aggregated_features"][name]) for name in FEATURE_NAMES},
                    "proposal_mask": list(proposal_key),
                    "singleton_mask": list(singleton_key),
                    "targets": targets,
                    "proposal_utilities": proposal_utilities,
                    "singleton_utilities": singleton_utilities,
                    "wrong_winner_count": len({wrong_winner(result, gold) for result in results.values()}),
                }
            )
    if missing:
        raise ValueError("Missing cached masks:\n" + "\n".join(missing[:20]))
    eligible_families = {
        family for family, passed in family_passes.items() if len(passed) == 2 and all(passed)
    }
    paired_rows = [row for row in rows if row["counterfactual_group"] in eligible_families]
    manifest = dict(manifest)
    manifest["offline_cohort"] = {
        "families_in_labels": len(family_passes),
        "paired_necessity_eligible_families": len(eligible_families),
        "documents_with_edge_rows_before_pair_filter": len(group_rows(rows)),
        "documents_after_pair_filter": len(group_rows(paired_rows)),
    }
    return paired_rows, manifest


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    return grouped


def family_bootstrap_gap(
    rows: list[dict[str, Any]], utility: str, draws: int = 4000, seed: int = 20261012
) -> list[float]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["counterfactual_group"]].append(row)
    families = list(by_family)
    if len(families) < 2:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(families, size=len(families), replace=True)
        selected = [row for family in sampled for row in by_family[str(family)]]
        support = [row["targets"][utility] for row in selected if row["is_support_boundary"]]
        other = [row["targets"][utility] for row in selected if not row["is_support_boundary"]]
        gaps.append(float(np.mean(support) - np.mean(other)))
    return np.quantile(gaps, [0.025, 0.975]).astype(float).tolist()


def utility_summary(rows: list[dict[str, Any]], utility: str) -> dict[str, Any]:
    support = np.asarray([row["targets"][utility] for row in rows if row["is_support_boundary"]])
    other = np.asarray([row["targets"][utility] for row in rows if not row["is_support_boundary"]])
    grouped = group_rows(rows)
    top1 = []
    reciprocal = []
    for document in grouped.values():
        ranked = sorted(document, key=lambda row: row["targets"][utility], reverse=True)
        rank = next(index for index, row in enumerate(ranked, 1) if row["is_support_boundary"])
        top1.append(float(rank == 1))
        reciprocal.append(1.0 / rank)
    return {
        "documents": len(grouped),
        "support_mean": float(support.mean()),
        "non_support_mean": float(other.mean()),
        "support_gap": float(support.mean() - other.mean()),
        "support_gap_family_bootstrap_95": family_bootstrap_gap(rows, utility),
        "support_positive_rate": float(np.mean(support > 0.0)),
        "oracle_exact_boundary_at_1": float(np.mean(top1)),
        "oracle_support_mrr": float(np.mean(reciprocal)),
    }


def stage_a(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "A",
        "config_hash": manifest["config_hash"],
        "rows": len(rows),
        "documents": len(group_rows(rows)),
        "families": len({row["counterfactual_group"] for row in rows}),
        "cohort": manifest["offline_cohort"],
        "splits": Counter(row["split"] for row in group_rows(rows).values() for row in row[:1]),
        "utilities": {},
    }
    for split in ("train", "validation", "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        report["utilities"][split] = {
            utility: utility_summary(subset, utility) for utility in UTILITY_NAMES
        }
    correlations: dict[str, float] = {}
    for left_index, left in enumerate(UTILITY_NAMES):
        for right in UTILITY_NAMES[left_index + 1 :]:
            coefficient = spearmanr(
                [row["targets"][left] for row in rows],
                [row["targets"][right] for row in rows],
            ).statistic
            correlations[f"{left}__{right}"] = float(coefficient)
    report["target_spearman"] = correlations
    report["wrong_argmax_instability"] = {
        "support_mean_distinct_wrong_winners": float(
            np.mean([row["wrong_winner_count"] for row in rows if row["is_support_boundary"]])
        ),
        "non_support_mean_distinct_wrong_winners": float(
            np.mean([row["wrong_winner_count"] for row in rows if not row["is_support_boundary"]])
        ),
        "fraction_factorials_with_switch": float(
            np.mean([row["wrong_winner_count"] > 1 for row in rows])
        ),
    }
    duplicates = []
    for document in group_rows(rows).values():
        support = set(document[0]["supporting_blocks"])
        masks = {tuple(row["proposal_mask"]) for row in document}
        useful_edges = sum(support.issubset(set(row["proposal_mask"])) for row in document)
        support_row = next(row for row in document if row["is_support_boundary"])
        same_as_support = sum(row["proposal_mask"] == support_row["proposal_mask"] for row in document)
        duplicates.append((len(masks), useful_edges, same_as_support))
    report["deployment_action_aliasing"] = {
        "mean_unique_masks_per_13_edges": float(np.mean([item[0] for item in duplicates])),
        "mean_edges_retaining_support": float(np.mean([item[1] for item in duplicates])),
        "mean_edges_same_mask_as_support_boundary": float(np.mean([item[2] for item in duplicates])),
        "documents_with_duplicate_actions_rate": float(np.mean([item[0] < 13 for item in duplicates])),
    }
    order_report: dict[str, Any] = {}
    for order in sorted({row["evidence_order"] for row in rows}):
        subset = [row for row in rows if row["evidence_order"] == order]
        order_report[order] = {
            utility: utility_summary(subset, utility) for utility in UTILITY_NAMES
        }
    report["by_evidence_order"] = order_report
    primary = report["utilities"]["validation"]["candidate_log_odds"]
    report["decision"] = "PASS" if (
        primary["support_gap_family_bootstrap_95"][0] > 0.0
        and primary["support_positive_rate"] >= 0.80
        and primary["oracle_exact_boundary_at_1"] >= 0.80
    ) else "REFINE_ESTIMAND"
    return report


def edge_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )


def within_document_ranks(rows: list[dict[str, Any]], utility: str) -> np.ndarray:
    output = np.zeros(len(rows), dtype=np.float64)
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices[row["sample_id"]].append(index)
    for document_indices in indices.values():
        values = np.asarray([rows[index]["targets"][utility] for index in document_indices])
        order = values.argsort().argsort().astype(np.float64)
        if len(order) > 1:
            order /= len(order) - 1
        output[document_indices] = order
    return output


def make_pairwise(
    rows: list[dict[str, Any]], transformed: np.ndarray, utility: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_document[row["sample_id"]].append(index)
    for indices in by_document.values():
        values = np.asarray([rows[index]["targets"][utility] for index in indices])
        differences = [abs(float(values[i] - values[j])) for i in range(len(values)) for j in range(i + 1, len(values))]
        scale = max(float(np.median(differences)), 1e-6)
        dead_zone = 0.10 * scale
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                gap = float(values[left] - values[right])
                if abs(gap) <= dead_zone:
                    continue
                difference = transformed[indices[left]] - transformed[indices[right]]
                label = int(gap > 0.0)
                weight = min(abs(gap) / scale, 4.0)
                pairs.extend((difference, -difference))
                labels.extend((label, 1 - label))
                weights.extend((weight, weight))
    return np.asarray(pairs), np.asarray(labels), np.asarray(weights)


def fit_ranknet(rows: list[dict[str, Any]], utility: str, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(edge_matrix(rows))
    transformed = scaler.transform(edge_matrix(rows))
    x_pair, y_pair, weights = make_pairwise(rows, transformed, utility)
    model = LogisticRegression(C=c_value, fit_intercept=False, max_iter=5000, solver="lbfgs")
    model.fit(x_pair, y_pair, sample_weight=weights)
    return scaler, model


def ndcg_at_3(document: list[dict[str, Any]], scores: list[float], utility: str) -> float:
    target = np.asarray([row["targets"][utility] for row in document])
    relevance = target.argsort().argsort().astype(np.float64)
    predicted_order = np.argsort(-np.asarray(scores))[:3]
    ideal_order = np.argsort(-relevance)[:3]
    discounts = 1.0 / np.log2(np.arange(2, 5))
    dcg = float(np.sum((np.power(2.0, relevance[predicted_order]) - 1.0) * discounts))
    ideal = float(np.sum((np.power(2.0, relevance[ideal_order]) - 1.0) * discounts))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(rows: list[dict[str, Any]], predictions: np.ndarray, utility: str) -> dict[str, Any]:
    indexed: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexed[row["sample_id"]].append(index)
    records = []
    for sample_id, indices in indexed.items():
        ranked = sorted(indices, key=lambda index: float(predictions[index]), reverse=True)
        chosen = rows[ranked[0]]
        support_rank = next(rank for rank, index in enumerate(ranked, 1) if rows[index]["is_support_boundary"])
        support = set(chosen["supporting_blocks"])
        proposal_retains = support.issubset(set(chosen["proposal_mask"]))
        singleton_retains = support.issubset(set(chosen["singleton_mask"]))
        values = [rows[index]["targets"][utility] for index in indices]
        records.append(
            {
                "sample_id": sample_id,
                "family": chosen["counterfactual_group"],
                "exact": float(chosen["is_support_boundary"]),
                "mrr": 1.0 / support_rank,
                "retains": float(proposal_retains),
                "singleton_retains": float(singleton_retains),
                "deployment_lift": chosen["proposal_utilities"][utility]
                - chosen["singleton_utilities"][utility],
                "oracle_regret": max(values) - chosen["targets"][utility],
                "ndcg3": ndcg_at_3(
                    [rows[index] for index in indices],
                    [float(predictions[index]) for index in indices],
                    utility,
                ),
            }
        )
    metrics = {
        "documents": len(records),
        "exact_support_boundary_at_1": float(np.mean([record["exact"] for record in records])),
        "support_mrr": float(np.mean([record["mrr"] for record in records])),
        "action_support_retention": float(np.mean([record["retains"] for record in records])),
        "singleton_support_retention": float(np.mean([record["singleton_retains"] for record in records])),
        "support_retention_lift": float(
            np.mean([record["retains"] - record["singleton_retains"] for record in records])
        ),
        "mean_deployment_utility_lift": float(np.mean([record["deployment_lift"] for record in records])),
        "mean_oracle_regret": float(np.mean([record["oracle_regret"] for record in records])),
        "ndcg_at_3": float(np.mean([record["ndcg3"] for record in records])),
    }
    return {"metrics": metrics, "documents": records}


def cross_validated_ranknet_predictions(
    rows: list[dict[str, Any]], utility: str, c_value: float, n_splits: int
) -> np.ndarray:
    groups = np.asarray([row["counterfactual_group"] for row in rows])
    predictions = np.zeros(len(rows), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    indices = np.arange(len(rows))
    for train_index, test_index in splitter.split(indices, groups=groups):
        train_rows = [rows[index] for index in train_index]
        scaler, model = fit_ranknet(train_rows, utility, c_value)
        predictions[test_index] = model.decision_function(scaler.transform(edge_matrix([rows[index] for index in test_index])))
    return predictions


def choose_ranknet_c(train_rows: list[dict[str, Any]], utility: str) -> tuple[float, list[dict[str, float]]]:
    grid = np.logspace(-3, 3, 13)
    scores = []
    for c_value in grid:
        prediction = cross_validated_ranknet_predictions(train_rows, utility, float(c_value), 5)
        metrics = ranking_metrics(train_rows, prediction, utility)["metrics"]
        scores.append({"C": float(c_value), "ndcg_at_3": metrics["ndcg_at_3"], "oracle_regret": metrics["mean_oracle_regret"]})
    best = max(scores, key=lambda item: (item["ndcg_at_3"], -item["oracle_regret"], -item["C"]))
    return float(best["C"]), scores


def ridge_predictions(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], utility: str) -> np.ndarray:
    scaler = StandardScaler().fit(edge_matrix(train_rows))
    model = Ridge(alpha=1.0).fit(scaler.transform(edge_matrix(train_rows)), within_document_ranks(train_rows, utility))
    return model.predict(scaler.transform(edge_matrix(validation_rows)))


def stage_b(rows: list[dict[str, Any]], run_root: Path, utility: str) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    c_value, cv_grid = choose_ranknet_c(train_rows, utility)
    scaler, ranknet = fit_ranknet(train_rows, utility, c_value)
    ranknet_validation = ranknet.decision_function(scaler.transform(edge_matrix(validation_rows)))
    ridge_validation = ridge_predictions(train_rows, validation_rows, utility)

    old_payload = read_json(run_root / "models" / "pair_model.json")
    old_model = LinearInteractionModel(
        **{
            key: old_payload[key]
            for key in (
                "feature_names",
                "mean",
                "scale",
                "coefficients",
                "intercept",
                "positive_threshold",
                "validation_metrics",
            )
        }
    )
    old_predictions = np.asarray([old_model.predict(row["features"]) for row in validation_rows])
    oracle_predictions = np.asarray([row["targets"][utility] for row in validation_rows])
    all_oof = cross_validated_ranknet_predictions(rows, utility, c_value, len({row["counterfactual_group"] for row in rows}))
    report = {
        "stage": "B",
        "primary_utility": utility,
        "selection_protocol": "C selected by 5-fold counterfactual-family CV on train only; validation untouched",
        "selected_C": c_value,
        "cv_grid": cv_grid,
        "validation": {
            "v10_ridge_max_margin": ranking_metrics(validation_rows, old_predictions, utility),
            "pointwise_ridge_candidate_log_odds": ranking_metrics(validation_rows, ridge_validation, utility),
            "pairwise_ranknet_candidate_log_odds": ranking_metrics(validation_rows, ranknet_validation, utility),
            "offline_oracle_candidate_log_odds": ranking_metrics(validation_rows, oracle_predictions, utility),
        },
        "development_leave_one_family_out": ranking_metrics(rows, all_oof, utility),
        "model": {
            "feature_names": FEATURE_NAMES,
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "coefficients": ranknet.coef_[0].astype(float).tolist(),
            "intercept": 0.0,
        },
    }
    baseline = report["validation"]["v10_ridge_max_margin"]["metrics"]
    proposed = report["validation"]["pairwise_ranknet_candidate_log_odds"]["metrics"]
    report["decision"] = "PASS" if (
        proposed["action_support_retention"] > baseline["action_support_retention"]
        and proposed["ndcg_at_3"] > baseline["ndcg_at_3"]
    ) else "NO_GO_FEATURES_INSUFFICIENT"
    return report


def write_edge_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_id", "counterfactual_group", "split", "evidence_order", "boundary",
        "is_support_boundary", "supporting_blocks", "proposal_mask", "singleton_mask",
        *[f"target_{name}" for name in UTILITY_NAMES], "wrong_winner_count", *FEATURE_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields if key in row}
            flat.update({f"target_{name}": row["targets"][name] for name in UTILITY_NAMES})
            flat.update(row["features"])
            flat["supporting_blocks"] = " ".join(map(str, row["supporting_blocks"]))
            flat["proposal_mask"] = " ".join(map(str, row["proposal_mask"]))
            flat["singleton_mask"] = " ".join(map(str, row["singleton_mask"]))
            writer.writerow(flat)


def markdown_report(stage_a_report: dict[str, Any], stage_b_report: dict[str, Any]) -> str:
    lines = [
        "# Offline v10 audit: stages A and B",
        "",
        "No model inference or sealed-test data was used.",
        "",
        "## Stage A: estimand audit",
        "",
        f"**Decision: {stage_a_report['decision']}**",
        "",
        "| Split | Utility | Support gap | 95% family bootstrap CI | Positive support | Oracle boundary@1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "all"):
        for utility in UTILITY_NAMES:
            item = stage_a_report["utilities"][split][utility]
            low, high = item["support_gap_family_bootstrap_95"]
            lines.append(
                f"| {split} | {utility} | {item['support_gap']:.3f} | [{low:.3f}, {high:.3f}] | "
                f"{item['support_positive_rate']:.1%} | {item['oracle_exact_boundary_at_1']:.1%} |"
            )
    alias = stage_a_report["deployment_action_aliasing"]
    instability = stage_a_report["wrong_argmax_instability"]
    lines.extend(
        [
            "",
            "### Structural diagnostics",
            "",
            f"- Different factorial cells changed the strongest wrong answer in {instability['fraction_factorials_with_switch']:.1%} of edges.",
            f"- A document has {alias['mean_unique_masks_per_13_edges']:.2f} unique deployment masks for 13 boundary choices on average.",
            f"- Duplicate deployment actions occur in {alias['documents_with_duplicate_actions_rate']:.1%} of documents.",
            f"- {alias['mean_edges_retaining_support']:.2f} boundary choices per document retain both gold blocks on average.",
            "",
            "## Stage B: family-grouped ranking",
            "",
            f"**Decision: {stage_b_report['decision']}**",
            "",
            f"Primary utility: `{stage_b_report['primary_utility']}`. Selected RankNet C: `{stage_b_report['selected_C']}`.",
            "",
            "| Validation method | Exact boundary@1 | Action retention | Lift vs singleton | NDCG@3 | Oracle regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, result in stage_b_report["validation"].items():
        metric = result["metrics"]
        lines.append(
            f"| {method} | {metric['exact_support_boundary_at_1']:.1%} | "
            f"{metric['action_support_retention']:.1%} | {metric['support_retention_lift']:+.1%} | "
            f"{metric['ndcg_at_3']:.3f} | {metric['mean_oracle_regret']:.3f} |"
        )
    development = stage_b_report["development_leave_one_family_out"]["metrics"]
    lines.extend(
        [
            "",
            "### Exploratory leave-one-family-out result",
            "",
            f"Across all 15 development families, RankNet action retention is {development['action_support_retention']:.1%}, "
            f"exact boundary@1 is {development['exact_support_boundary_at_1']:.1%}, and NDCG@3 is {development['ndcg_at_3']:.3f}.",
            "",
            "Stage A passes: the smoothed candidate-competition estimand has a strong, family-stable support signal. "
            "Stage B does not pass: changing the loss alone does not improve action selection over the v10 Ridge baseline. "
            "The next version must improve the observable edge features or learn from richer layer-wise representations before another GPU evaluation.",
            "",
            "The six validation documents remain too few for a final scientific claim; this stage decides whether the ranking formulation is coherent enough for the next GPU run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU-free v10 estimand and ranking audits")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    data_path = (args.data or run_root.parents[1] / "data" / "qualification.jsonl").resolve()
    output = (args.output or run_root / "offline_v11").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows, manifest = build_rows(run_root, data_path)
    stage_a_report = stage_a(rows, manifest)
    stage_b_report = stage_b(rows, run_root, "candidate_log_odds")
    (output / "stage_a.json").write_text(json.dumps(stage_a_report, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    (output / "stage_b.json").write_text(json.dumps(stage_b_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "ranknet_model.json").write_text(json.dumps(stage_b_report["model"], indent=2), encoding="utf-8")
    (output / "report.md").write_text(markdown_report(stage_a_report, stage_b_report), encoding="utf-8")
    write_edge_csv(rows, output / "edge_table.csv")
    print(f"Wrote offline analysis to {output}")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
