from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bikv.features import FEATURE_NAMES, feature_vector


@dataclass
class LinearInteractionModel:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    coefficients: list[float]
    intercept: float
    positive_threshold: float = 0.0
    validation_metrics: dict[str, float] | None = None

    def predict(self, row: dict[str, float]) -> float:
        values = feature_vector(row)
        standardized = [
            (value - mean) / scale if abs(scale) > 1e-12 else 0.0
            for value, mean, scale in zip(values, self.mean, self.scale)
        ]
        return float(self.intercept + sum(coef * value for coef, value in zip(self.coefficients, standardized)))

    def probability(self, row: dict[str, float]) -> float:
        centered = self.predict(row) - self.positive_threshold
        centered = max(-30.0, min(30.0, centered))
        return 1.0 / (1.0 + math.exp(-centered))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "LinearInteractionModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _within_sample_rank_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    targets = np.zeros(len(rows), dtype=np.float64)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["sample_id"]), []).append(index)
    for indices in grouped.values():
        values = np.asarray([float(rows[index]["target"]) for index in indices], dtype=np.float64)
        unique = np.unique(values)
        if unique.size > 1:
            rank_by_value = {float(value): rank / (unique.size - 1) for rank, value in enumerate(unique)}
            ranks = np.asarray([rank_by_value[float(value)] for value in values], dtype=np.float64)
        else:
            ranks = np.zeros(len(indices), dtype=np.float64)
        targets[indices] = ranks
    return targets


def _ranking_metrics(rows: list[dict[str, Any]], prediction: np.ndarray, prefix: str) -> dict[str, float]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["sample_id"]), []).append(index)
    reciprocal_ranks = []
    exact_pair_hits = []
    support_retention = []
    singleton_retention = []
    utility_lifts = []
    for indices in grouped.values():
        ranked = sorted(indices, key=lambda index: float(prediction[index]), reverse=True)
        chosen = rows[ranked[0]]
        support_index = next(
            (rank for rank, index in enumerate(ranked, start=1) if rows[index].get("is_support_boundary")),
            None,
        )
        if support_index is not None:
            reciprocal_ranks.append(1.0 / support_index)
        exact_pair_hits.append(float(bool(chosen.get("is_support_boundary"))))
        support = set(int(value) for value in chosen.get("supporting_blocks", []))
        proposal = set(int(value) for value in chosen.get("proposal_mask", []))
        singleton = set(int(value) for value in chosen.get("singleton_mask", []))
        support_retention.append(float(support.issubset(proposal)))
        singleton_retention.append(float(support.issubset(singleton)))
        utility_lifts.append(
            float(chosen.get("proposal_utility", chosen["target"]))
            - float(chosen.get("singleton_utility", 0.0))
        )
    support_interactions = [float(row["target"]) for row in rows if row.get("is_support_boundary")]
    other_interactions = [float(row["target"]) for row in rows if not row.get("is_support_boundary")]
    metrics = {
        f"{prefix}_documents": float(len(grouped)),
        f"{prefix}_pair_recall_at_1": float(np.mean(exact_pair_hits)) if exact_pair_hits else 0.0,
        f"{prefix}_support_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        f"{prefix}_support_retention": float(np.mean(support_retention)) if support_retention else 0.0,
        f"{prefix}_singleton_support_retention": (
            float(np.mean(singleton_retention)) if singleton_retention else 0.0
        ),
        f"{prefix}_support_retention_lift": (
            float(np.mean(support_retention) - np.mean(singleton_retention)) if support_retention else 0.0
        ),
        f"{prefix}_mean_margin_lift_over_singleton": (
            float(np.mean(utility_lifts)) if utility_lifts else 0.0
        ),
        f"{prefix}_support_mean_interaction": (
            float(np.mean(support_interactions)) if support_interactions else 0.0
        ),
        f"{prefix}_non_support_mean_interaction": (
            float(np.mean(other_interactions)) if other_interactions else 0.0
        ),
        f"{prefix}_support_interaction_gap": (
            float(np.mean(support_interactions) - np.mean(other_interactions))
            if support_interactions and other_interactions
            else 0.0
        ),
        f"{prefix}_support_positive_interaction_rate": (
            float(np.mean(np.asarray(support_interactions) > 0.0)) if support_interactions else 0.0
        ),
    }
    return metrics


def _document_bootstrap_support_gap(
    rows: list[dict[str, Any]], draws: int = 1000, seed: int = 1701
) -> tuple[float, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    documents = list(grouped.values())
    if len(documents) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        sampled = [documents[index] for index in rng.integers(0, len(documents), len(documents))]
        support = [float(row["target"]) for document in sampled for row in document if row.get("is_support_boundary")]
        other = [float(row["target"]) for document in sampled for row in document if not row.get("is_support_boundary")]
        if support and other:
            gaps.append(float(np.mean(support) - np.mean(other)))
    if not gaps:
        return 0.0, 0.0
    low, high = np.quantile(np.asarray(gaps), [0.025, 0.975])
    return float(low), float(high)


def train_linear_predictor(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> LinearInteractionModel:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not train_rows:
        raise ValueError("No training rows were found")
    x_train = np.asarray([feature_vector(row["features"]) for row in train_rows], dtype=np.float64)
    y_train = _within_sample_rank_targets(train_rows)
    groups = np.asarray([str(row.get("counterfactual_group", row["story_id"])) for row in train_rows])
    unique_groups = np.unique(groups)
    alphas = np.logspace(-4, 4, 25)
    if unique_groups.size >= 2:
        search = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": alphas},
            cv=GroupKFold(n_splits=min(5, unique_groups.size)),
            scoring="neg_mean_squared_error",
        )
        search.fit(x_train, y_train, groups=groups)
        scaler = search.best_estimator_.named_steps["scaler"]
        regressor = search.best_estimator_.named_steps["ridge"]
    else:
        scaler = StandardScaler().fit(x_train)
        regressor = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    x_scaled = scaler.transform(x_train)
    positive_threshold = 0.5

    metrics: dict[str, float] = {
        "n_train": float(len(train_rows)),
        "train_pearson": _correlation(regressor.predict(x_scaled), y_train),
        "ridge_alpha": float(regressor.alpha),
        "train_counterfactual_groups": float(unique_groups.size),
    }
    metrics.update(_ranking_metrics(train_rows, regressor.predict(x_scaled), "train"))
    if validation_rows:
        x_validation = np.asarray([feature_vector(row["features"]) for row in validation_rows], dtype=np.float64)
        y_validation = _within_sample_rank_targets(validation_rows)
        prediction = regressor.predict(scaler.transform(x_validation))
        metrics.update(
            {
                "n_validation": float(len(validation_rows)),
                "validation_pearson": _correlation(prediction, y_validation),
                "validation_mae": float(np.mean(np.abs(prediction - y_validation))),
                "validation_above_median_accuracy": float(
                    np.mean((prediction > positive_threshold) == (y_validation > positive_threshold))
                ),
            }
        )
        metrics.update(_ranking_metrics(validation_rows, prediction, "validation"))
        gap_low, gap_high = _document_bootstrap_support_gap(validation_rows)
        metrics["validation_support_interaction_gap_bootstrap_low"] = gap_low
        metrics["validation_support_interaction_gap_bootstrap_high"] = gap_high

    return LinearInteractionModel(
        feature_names=list(FEATURE_NAMES),
        mean=scaler.mean_.astype(float).tolist(),
        scale=scaler.scale_.astype(float).tolist(),
        coefficients=np.asarray(regressor.coef_).astype(float).tolist(),
        intercept=float(regressor.intercept_),
        positive_threshold=positive_threshold,
        validation_metrics=metrics,
    )
