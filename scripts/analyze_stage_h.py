from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable


SYNTHETIC_METHODS = (
    "full",
    "task_oracle",
    "relation_graph",
    "graph_matched_unary_ridge_v1",
    "graph_arbitrary_pair_ridge_v1",
    "graph_pair_shuffled_v1",
    "snapkv_upstream_query_aware",
)
RULER_METHODS = (
    "full",
    "relation_graph",
    "graph_matched_unary_ridge_v1",
    "graph_arbitrary_pair_ridge_v1",
    "graph_pair_shuffled_v1",
    "snapkv_upstream_query_aware",
)


def rows(root: Path, namespace: str, methods: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for method in methods:
        directory = root / "stage_h" / namespace / method
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("method") != method:
                raise ValueError(f"Method mismatch in {path}")
            result.append(row)
    return result


def family_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    seed: int,
    repeats: int = 5000,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    families: dict[str, list[str]] = defaultdict(list)
    for sample_id in common:
        families[str(left[sample_id]["counterfactual_group"])].append(sample_id)
    family_names = sorted(families)
    if not family_names:
        return {"difference": None, "ci95": None, "families": 0}
    family_delta = {
        family: mean(value(left[sample]) - value(right[sample]) for sample in sample_ids)
        for family, sample_ids in families.items()
    }
    observed = mean(family_delta.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        draws.append(mean(family_delta[rng.choice(family_names)] for _ in family_names))
    draws.sort()
    return {
        "difference": observed,
        "ci95": [draws[int(0.025 * repeats)], draws[int(0.975 * repeats) - 1]],
        "families": len(family_names),
    }


def validate(rows_: list[dict[str, Any]], budget: int) -> None:
    for row in rows_:
        selected = row.get("selected_blocks")
        if selected is None:
            continue
        if len(selected) != budget or len(set(selected)) != budget:
            raise ValueError(f"Invalid budget or duplicate mask for {row['sample_id']} {row['method']}")
        if min(selected) < 0:
            raise ValueError(f"Negative block index for {row['sample_id']} {row['method']}")


def synthetic_report(rows_: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        grouped[row["method"]].append(row)
    summary = {}
    lines = [
        "## Controlled chains (4k)", "",
        "| Method | N | Candidate accuracy | Greedy accuracy | Complete-chain retention | Mean support fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in SYNTHETIC_METHODS:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        support_rows = [row for row in method_rows if row.get("support_retained") is not None]
        item = {
            "n": len(method_rows),
            "candidate_accuracy": mean(float(row["candidate_result"]["is_correct"]) for row in method_rows),
            "greedy_accuracy": mean(float(row["generation_result"]["is_correct"]) for row in method_rows),
            "complete_chain_retention": mean(float(row["support_retained"]) for row in support_rows) if support_rows else None,
            "mean_support_fraction": mean(float(row["support_fraction"]) for row in support_rows) if support_rows else None,
        }
        summary[method] = item
        retention = "n/a" if item["complete_chain_retention"] is None else f"{item['complete_chain_retention']:.1%}"
        fraction = "n/a" if item["mean_support_fraction"] is None else f"{item['mean_support_fraction']:.1%}"
        lines.append(
            f"| {method} | {item['n']} | {item['candidate_accuracy']:.1%} | "
            f"{item['greedy_accuracy']:.1%} | {retention} | {fraction} |"
        )
    indices = {method: {row["sample_id"]: row for row in values} for method, values in grouped.items()}
    comparisons = {}
    if "graph_arbitrary_pair_ridge_v1" in indices and "graph_matched_unary_ridge_v1" in indices:
        comparisons["pair_minus_matched_unary_accuracy"] = family_bootstrap(
            indices["graph_arbitrary_pair_ridge_v1"], indices["graph_matched_unary_ridge_v1"],
            lambda row: float(row["candidate_result"]["is_correct"]), seed,
        )
        comparisons["pair_minus_matched_unary_retention"] = family_bootstrap(
            indices["graph_arbitrary_pair_ridge_v1"], indices["graph_matched_unary_ridge_v1"],
            lambda row: float(row["support_retained"]), seed + 1,
        )
    lines.extend(["", "Primary matched comparisons:", "", "```json", json.dumps(comparisons, indent=2), "```"])
    strata = {}
    for key in ((2, "adjacent"), (2, "dispersed"), (3, "adjacent"), (3, "dispersed"), (4, "adjacent"), (4, "dispersed")):
        label = f"h{key[0]}_{key[1]}"
        strata[label] = {}
        for method, values in grouped.items():
            subset = [row for row in values if int(row["chain_length"]) == key[0] and row["evidence_condition"] == key[1]]
            if subset:
                strata[label][method] = mean(float(row["candidate_result"]["is_correct"]) for row in subset)
    return {"summary": summary, "comparisons": comparisons, "strata_candidate_accuracy": strata}, lines


def ruler_report(rows_: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        grouped[row["method"]].append(row)
    summary = {}
    lines = [
        "## RULER Variable Tracking (4k subset)", "",
        "| Method | N | Reference recall | Set F1 | Set EM |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in RULER_METHODS:
        values = grouped.get(method, [])
        if not values:
            continue
        item = {
            "n": len(values),
            "reference_recall": mean(float(row["ruler_metrics"]["reference_recall"]) for row in values),
            "set_f1": mean(float(row["ruler_metrics"]["set_f1"]) for row in values),
            "set_em": mean(float(row["ruler_metrics"]["set_exact_match"]) for row in values),
        }
        summary[method] = item
        lines.append(
            f"| {method} | {item['n']} | {item['reference_recall']:.1%} | "
            f"{item['set_f1']:.1%} | {item['set_em']:.1%} |"
        )
    return summary, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage H arbitrary-pair and chain runs")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()
    root = args.run_root.resolve()
    synthetic = rows(root, "synthetic_4k", SYNTHETIC_METHODS)
    ruler = rows(root, "ruler_4k", RULER_METHODS)
    validate(synthetic + ruler, args.budget)
    report: dict[str, Any] = {"stage": "H", "integrity": "PASS", "exploratory": True}
    text = ["# Stage H arbitrary-pair and chain analysis", "", "Integrity: PASS", ""]
    if synthetic:
        report["synthetic"], lines = synthetic_report(synthetic, args.seed)
        text.extend(lines + [""])
    if ruler:
        report["ruler"] , lines = ruler_report(ruler)
        text.extend(lines + [""])
    text.append(
        "Stage H is an exploratory extension. Relation-graph results diagnose explicit chain connectivity and "
        "must not be reported as the learned BIKV pair model."
    )
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
