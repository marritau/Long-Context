from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PRIMARY = "candidate_pair_no_query_v2"
MATCHED_UNARY = "candidate_matched_unary_ridge_v2"


def read_rows(root: Path, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob(pattern))]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_family_bootstrap(
    left: dict[str, dict], right: dict[str, dict], sample_ids: set[str], metric: Callable[[dict], float], seed: int
) -> dict[str, Any]:
    common = sorted(sample_ids & set(left) & set(right))
    groups: dict[str, list[float]] = defaultdict(list)
    for sample_id in common:
        family = str(left[sample_id]["counterfactual_group"])
        groups[family].append(metric(left[sample_id]) - metric(right[sample_id]))
    family_values = np.asarray([np.mean(values) for values in groups.values()], dtype=np.float64)
    if not len(family_values):
        return {"samples": 0, "families": 0, "mean_difference": None, "family_bootstrap_95": None}
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(family_values, size=len(family_values), replace=True).mean() for _ in range(10000)
    ])
    return {
        "samples": len(common),
        "families": len(family_values),
        "mean_difference": float(family_values.mean()),
        "family_bootstrap_95": percentile_interval(draws),
    }


def summarize_g1b(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    grouped: dict[tuple[int, str, int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (int(row["n_blocks"]), str(row["arm"]), int(row["budget_blocks"]), str(row["method"]))
        grouped[key][str(row["sample_id"])] = row
    output: dict[str, Any] = {"groups": {}, "comparisons": {}}
    full_ids: dict[int, set[str]] = {}
    for (n_blocks, arm, budget, method), values in sorted(grouped.items()):
        if method == "full" and arm == "fixed":
            full_ids[n_blocks] = {
                sample_id for sample_id, row in values.items() if bool(row["candidate_result"]["is_correct"])
            }
    for (n_blocks, arm, budget, method), values in sorted(grouped.items()):
        cohort = full_ids.get(n_blocks, set(values)) & set(values)
        selected = [values[sample_id] for sample_id in sorted(cohort)]
        support = [float(row["support_retained"]) for row in selected if row["support_retained"] is not None]
        key = f"n{n_blocks}_{arm}_b{budget}_{method}"
        output["groups"][key] = {
            "samples": len(values),
            "full_correct_cohort": len(selected),
            "all_sample_candidate_accuracy": float(np.mean([
                row["candidate_result"]["is_correct"] for row in values.values()
            ])) if values else None,
            "all_sample_greedy_exact_match": float(np.mean([
                row["generation_result"]["is_correct"] for row in values.values()
            ])) if values else None,
            "candidate_accuracy": float(np.mean([row["candidate_result"]["is_correct"] for row in selected])) if selected else None,
            "greedy_exact_match": float(np.mean([row["generation_result"]["is_correct"] for row in selected])) if selected else None,
            "candidate_margin": float(np.mean([row["candidate_result"]["gold_margin"] for row in selected])) if selected else None,
            "support_retention": float(np.mean(support)) if support else None,
            "adjacent_candidate_accuracy": float(np.mean([
                row["candidate_result"]["is_correct"] for row in selected if row["evidence_condition"] == "adjacent"
            ])) if any(row["evidence_condition"] == "adjacent" for row in selected) else None,
            "separated_candidate_accuracy": float(np.mean([
                row["candidate_result"]["is_correct"] for row in selected if row["evidence_condition"] == "separated"
            ])) if any(row["evidence_condition"] == "separated" for row in selected) else None,
        }
    comparison_keys = sorted({(n, arm, budget) for n, arm, budget, method in grouped if method == PRIMARY})
    for n_blocks, arm, budget in comparison_keys:
        pair = grouped[(n_blocks, arm, budget, PRIMARY)]
        unary = grouped.get((n_blocks, arm, budget, MATCHED_UNARY), {})
        cohort = full_ids.get(n_blocks, set(pair))
        adjacent = {sid for sid in cohort if pair.get(sid, {}).get("evidence_condition") == "adjacent"}
        key = f"n{n_blocks}_{arm}_b{budget}"
        output["comparisons"][key] = {
            "pair_vs_matched_unary_accuracy": paired_family_bootstrap(
                pair, unary, cohort, lambda row: float(row["candidate_result"]["is_correct"]), seed + n_blocks + budget
            ),
            "pair_vs_matched_unary_margin": paired_family_bootstrap(
                pair, unary, cohort, lambda row: float(row["candidate_result"]["gold_margin"]), seed + n_blocks + budget + 1
            ),
            "pair_vs_matched_unary_adjacent_accuracy": paired_family_bootstrap(
                pair, unary, adjacent, lambda row: float(row["candidate_result"]["is_correct"]), seed + n_blocks + budget + 2
            ),
            "mask_agreement": float(np.mean([
                pair[sid]["selected_blocks"] == unary[sid]["selected_blocks"]
                for sid in sorted(cohort & set(pair) & set(unary))
            ])) if cohort & set(pair) & set(unary) else None,
        }
        full_groups = [
            values for (n, arm_name, _full_budget, method), values in grouped.items()
            if n == n_blocks and arm_name == "fixed" and method == "full"
        ]
        full = full_groups[0] if full_groups else {}
        all_ids = set(pair) & set(full)
        output["comparisons"][key]["pair_vs_full_all_sample_accuracy"] = paired_family_bootstrap(
            pair,
            full,
            all_ids,
            lambda row: float(row["candidate_result"]["is_correct"]),
            seed + n_blocks + budget + 3,
        )
    return output


def summarize_ruler(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["length_label"]), str(row["method"]))][str(row["sample_id"])] = row
    output: dict[str, Any] = {"groups": {}, "comparisons": {}}
    full_exact: dict[int, set[str]] = {}
    for (length, method), values in grouped.items():
        if method == "full":
            full_exact[length] = {
                sid for sid, row in values.items() if bool(row["ruler_metrics"]["set_exact_match"])
            }
    for (length, method), values in sorted(grouped.items()):
        rows_here = list(values.values())
        conditional = [values[sid] for sid in sorted(full_exact.get(length, set()) & set(values))]
        output["groups"][f"length{length}_{method}"] = {
            "samples": len(rows_here),
            "reference_recall": float(np.mean([row["ruler_metrics"]["reference_recall"] for row in rows_here])),
            "set_f1": float(np.mean([row["ruler_metrics"]["set_f1"] for row in rows_here])),
            "set_exact_match": float(np.mean([row["ruler_metrics"]["set_exact_match"] for row in rows_here])),
            "full_exact_cohort": len(conditional),
            "conditional_set_exact_match": float(np.mean([
                row["ruler_metrics"]["set_exact_match"] for row in conditional
            ])) if conditional else None,
        }
    for length in sorted({key[0] for key in grouped if key[1] == PRIMARY}):
        pair = grouped[(length, PRIMARY)]
        unary = grouped.get((length, MATCHED_UNARY), {})
        ids = set(pair) & set(unary)
        differences = np.asarray([
            float(pair[sid]["ruler_metrics"]["set_f1"]) - float(unary[sid]["ruler_metrics"]["set_f1"])
            for sid in sorted(ids)
        ])
        rng = np.random.default_rng(seed + length)
        draws = np.asarray([
            rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)
        ]) if len(differences) else np.asarray([])
        output["comparisons"][str(length)] = {
            "samples": len(differences),
            "pair_minus_matched_unary_set_f1": float(differences.mean()) if len(differences) else None,
            "sample_bootstrap_95": percentile_interval(draws) if len(draws) else None,
        }
        full = grouped.get((length, "full"), {})
        rescue_ids = sorted(set(pair) & set(full))
        rescue_differences = np.asarray([
            float(pair[sid]["ruler_metrics"]["set_f1"])
            - float(full[sid]["ruler_metrics"]["set_f1"])
            for sid in rescue_ids
        ], dtype=np.float64)
        rescue_draws = np.asarray([
            rng.choice(rescue_differences, size=len(rescue_differences), replace=True).mean()
            for _ in range(10000)
        ]) if len(rescue_differences) else np.asarray([])
        output["comparisons"][str(length)].update({
            "pair_minus_full_set_f1": float(rescue_differences.mean()) if len(rescue_differences) else None,
            "pair_minus_full_bootstrap_95": percentile_interval(rescue_draws) if len(rescue_draws) else None,
        })
    return output


def validate_integrity(g1b: list[dict], ruler: list[dict]) -> list[str]:
    errors = []
    for row in g1b:
        mask = row["selected_blocks"]
        method = row["method"]
        if mask is not None and (len(mask) != row["budget_blocks"] or len(set(mask)) != len(mask)):
            errors.append(f"{row['sample_id']}/{method}: invalid mask")
        for result_name in ("candidate_result", "generation_result"):
            lengths = row[result_name].get("cache_lengths", [])
            if lengths and len(set(lengths)) != 1:
                errors.append(f"{row['sample_id']}/{method}: nonuniform {result_name} cache")
        if row.get("query_diagnostic_cache_reused") is not False:
            errors.append(f"{row['sample_id']}/{method}: diagnostic cache reuse not false")
    for row in ruler:
        mask = row["selected_blocks"]
        budget = row["budget_blocks"]
        if mask is not None and (budget is None or len(mask) != budget or len(set(mask)) != len(mask)):
            errors.append(f"{row['sample_id']}/{row['method']}: invalid RULER mask")
        lengths = row["generation_result"].get("cache_lengths", [])
        if lengths and len(set(lengths)) != 1:
            errors.append(f"{row['sample_id']}/{row['method']}: nonuniform RULER cache")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze combined Stage G1b and G2")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    root = args.run_root.resolve()
    g1b_rows = read_rows(root, "g1b/**/*.json")
    ruler_rows = read_rows(root, "g2_ruler/**/*.json")
    errors = validate_integrity(g1b_rows, ruler_rows)
    report = {
        "integrity": {"passed": not errors, "errors": errors},
        "g1b": summarize_g1b(g1b_rows, args.seed) if g1b_rows else None,
        "g2_ruler": summarize_ruler(ruler_rows, args.seed) if ruler_rows else None,
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage_g1b_g2_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Stage G1b + G2", "", f"Integrity: {'PASS' if not errors else 'FAIL'}", ""]
    if report["g1b"]:
        lines.extend(["## G1b", "", "| Group | All-sample accuracy | Full-correct accuracy | Greedy EM | Support retention |", "|---|---:|---:|---:|---:|"])
        for key, value in report["g1b"]["groups"].items():
            def pct(item):
                return "-" if item is None else f"{item:.1%}"
            lines.append(
                f"| {key} | {pct(value['all_sample_candidate_accuracy'])} | {pct(value['candidate_accuracy'])} | {pct(value['greedy_exact_match'])} | {pct(value['support_retention'])} |"
            )
        lines.append("")
        lines.extend([
            "### G1b primary comparisons",
            "",
            "| Group | Pair - matched unary accuracy | 95% family CI | Pair - full all-sample accuracy | 95% family CI |",
            "|---|---:|---:|---:|---:|",
        ])
        for key, value in report["g1b"]["comparisons"].items():
            matched = value["pair_vs_matched_unary_accuracy"]
            rescue = value["pair_vs_full_all_sample_accuracy"]
            def delta(item):
                return "-" if item["mean_difference"] is None else f"{item['mean_difference']:+.1%}"
            def interval(item):
                ci = item["family_bootstrap_95"]
                return "-" if ci is None else f"[{ci[0]:+.1%}, {ci[1]:+.1%}]"
            lines.append(f"| {key} | {delta(matched)} | {interval(matched)} | {delta(rescue)} | {interval(rescue)} |")
        lines.append("")
    if report["g2_ruler"]:
        lines.extend(["## RULER Variable Tracking", "", "| Group | Reference recall | Set F1 | Set EM |", "|---|---:|---:|---:|"])
        for key, value in report["g2_ruler"]["groups"].items():
            lines.append(
                f"| {key} | {value['reference_recall']:.1%} | {value['set_f1']:.1%} | {value['set_exact_match']:.1%} |"
            )
        lines.extend([
            "",
            "### RULER primary comparisons",
            "",
            "| Length | Pair - matched unary Set F1 | 95% CI | Pair - full Set F1 | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ])
        for length, value in report["g2_ruler"]["comparisons"].items():
            unary_ci = value["sample_bootstrap_95"]
            full_ci = value["pair_minus_full_bootstrap_95"]
            lines.append(
                f"| {length} | {value['pair_minus_matched_unary_set_f1']:+.1%} | "
                f"[{unary_ci[0]:+.1%}, {unary_ci[1]:+.1%}] | {value['pair_minus_full_set_f1']:+.1%} | "
                f"[{full_ci[0]:+.1%}, {full_ci[1]:+.1%}] |"
            )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))
    if errors:
        raise RuntimeError(f"Integrity validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
