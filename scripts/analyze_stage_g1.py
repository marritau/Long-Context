from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = [
    "random_block",
    "shared_block_knorm",
    "shared_block_query_attention",
    "candidate_query_unary_v2",
    "candidate_pair_no_query_v2",
    "task_oracle",
    "token_knorm",
    "snapkv_upstream_query_aware",
]
PRIMARY = "candidate_pair_no_query_v2"
UNARY = "candidate_query_unary_v2"
EXTERNAL_BASELINES = ["shared_block_knorm", "snapkv_upstream_query_aware"]


def load_budget(root: Path, budget: int) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for method in METHODS:
        directory = root / "evaluation" / f"budget_{budget:02d}" / method
        result[method] = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        }
    return result


def summarize(results, ids: list[str]) -> dict[str, Any]:
    output = {}
    for method in METHODS:
        rows = [results[method][sample_id] for sample_id in ids]
        support = [row["support_retained"] for row in rows if row["support_retained"] is not None]
        output[method] = {
            "n": len(rows),
            "candidate_accuracy": float(np.mean([row["candidate_result"]["is_correct"] for row in rows])),
            "greedy_accuracy": float(np.mean([row["generation_result"]["is_correct"] for row in rows])),
            "support_retention": None if not support else float(np.mean(support)),
            "mean_candidate_margin": float(np.mean([row["candidate_result"]["gold_margin"] for row in rows])),
            "measured_seconds": float(sum(
                row["candidate_result"]["elapsed_seconds"] + row["generation_result"]["elapsed_seconds"]
                for row in rows
            )),
        }
    return output


def paired_family_bootstrap(results, ids, proposed, baseline, seed=20261014, draws=20000):
    families: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        families[results[proposed][sample_id]["counterfactual_group"]].append(sample_id)
    names = sorted(families)

    def difference(sample_ids, key, cast=float):
        return float(np.mean([
            cast(results[proposed][sample_id][key]["is_correct"])
            - cast(results[baseline][sample_id][key]["is_correct"])
            for sample_id in sample_ids
        ]))

    observed = difference(ids, "candidate_result")
    margin_observed = float(np.mean([
        results[proposed][sample_id]["candidate_result"]["gold_margin"]
        - results[baseline][sample_id]["candidate_result"]["gold_margin"]
        for sample_id in ids
    ]))
    rng = np.random.default_rng(seed)
    accuracy_draws = []
    margin_draws = []
    for _ in range(draws):
        selected = rng.choice(names, size=len(names), replace=True)
        sample_ids = [sample_id for family in selected for sample_id in families[str(family)]]
        accuracy_draws.append(difference(sample_ids, "candidate_result"))
        margin_draws.append(float(np.mean([
            results[proposed][sample_id]["candidate_result"]["gold_margin"]
            - results[baseline][sample_id]["candidate_result"]["gold_margin"]
            for sample_id in sample_ids
        ])))
    proposed_only = sum(
        bool(results[proposed][sample_id]["candidate_result"]["is_correct"])
        and not bool(results[baseline][sample_id]["candidate_result"]["is_correct"])
        for sample_id in ids
    )
    baseline_only = sum(
        bool(results[baseline][sample_id]["candidate_result"]["is_correct"])
        and not bool(results[proposed][sample_id]["candidate_result"]["is_correct"])
        for sample_id in ids
    )
    discordant = proposed_only + baseline_only
    exact_p = 1.0 if discordant == 0 else min(
        1.0,
        2 * sum(math.comb(discordant, value) for value in range(min(proposed_only, baseline_only) + 1))
        / (2 ** discordant),
    )
    return {
        "candidate_accuracy_lift": observed,
        "candidate_accuracy_family_bootstrap_95": np.quantile(accuracy_draws, [0.025, 0.975]).astype(float).tolist(),
        "candidate_margin_lift": margin_observed,
        "candidate_margin_family_bootstrap_95": np.quantile(margin_draws, [0.025, 0.975]).astype(float).tolist(),
        "proposed_only_correct": proposed_only,
        "baseline_only_correct": baseline_only,
        "mcnemar_exact_two_sided_p": exact_p,
        "families": len(names),
    }


def mask_diagnostics(results, ids: list[str]) -> dict[str, Any]:
    equal = []
    jaccard = []
    pair_support_only = 0
    unary_support_only = 0
    for sample_id in ids:
        pair = set(results[PRIMARY][sample_id]["selected_blocks"])
        unary = set(results[UNARY][sample_id]["selected_blocks"])
        equal.append(pair == unary)
        jaccard.append(len(pair & unary) / len(pair | unary))
        supports = set(results[PRIMARY][sample_id]["supporting_blocks"])
        pair_kept = supports.issubset(pair)
        unary_kept = supports.issubset(unary)
        pair_support_only += pair_kept and not unary_kept
        unary_support_only += unary_kept and not pair_kept
    return {
        "exact_mask_agreement": float(np.mean(equal)),
        "mean_mask_jaccard": float(np.mean(jaccard)),
        "pair_only_retains_support": int(pair_support_only),
        "unary_only_retains_support": int(unary_support_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage G1a frozen-selector budget stress")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[6, 10, 18])
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE":
        failures.append(f"run is not complete: {status}")
    source_validation = json.loads((root / "source_validation.json").read_text(encoding="utf-8"))
    if not source_validation.get("passed"):
        failures.append("Stage F2 source validation did not pass")

    full_directory = root / "evaluation" / "budget_18" / "full"
    full = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in full_directory.glob("*.json")}
    ids = sorted(full)
    eligible = [sample_id for sample_id in ids if full[sample_id]["candidate_result"]["is_correct"]]
    budgets: dict[str, Any] = {}
    for budget in args.budgets:
        results = load_budget(root, budget)
        budget_failures = []
        for method in METHODS:
            if sorted(results[method]) != ids:
                budget_failures.append(f"B={budget} {method}: sample mismatch ({len(results[method])}/{len(ids)})")
        failures.extend(budget_failures)
        if budget_failures:
            continue
        protocols = {
            method: sorted({row["cache_protocol"] for row in results[method].values()})
            for method in METHODS
        }
        if protocols["snapkv_upstream_query_aware"] != ["query_in_compressed_prefill"]:
            failures.append(f"B={budget}: SnapKV protocol mismatch")
        if any(
            protocols[method] != ["document_compress_then_query"]
            for method in METHODS if method != "snapkv_upstream_query_aware"
        ):
            failures.append(f"B={budget}: BIKV/control protocol mismatch")
        metrics_all = summarize(results, ids)
        metrics_eligible = summarize(results, eligible)
        comparisons = {
            f"pair_vs_{baseline}_all": paired_family_bootstrap(results, ids, PRIMARY, baseline)
            for baseline in [UNARY, *EXTERNAL_BASELINES]
        }
        comparisons.update({
            f"pair_vs_{baseline}_full_correct": paired_family_bootstrap(results, eligible, PRIMARY, baseline)
            for baseline in [UNARY, *EXTERNAL_BASELINES]
        })
        budgets[str(budget)] = {
            "protocols": protocols,
            "cohorts": {"all": metrics_all, "full_context_candidate_correct": metrics_eligible},
            "comparisons": comparisons,
            "mask_diagnostics_full_correct": mask_diagnostics(results, eligible),
        }

    primary_budget = budgets.get("6")
    if failures or primary_budget is None:
        decision = "INVALID_OR_INCOMPLETE"
    else:
        eligible_metrics = primary_budget["cohorts"]["full_context_candidate_correct"]
        unary_comparison = primary_budget["comparisons"][f"pair_vs_{UNARY}_full_correct"]
        beats_knorm = (
            eligible_metrics[PRIMARY]["candidate_accuracy"]
            > eligible_metrics["shared_block_knorm"]["candidate_accuracy"]
        )
        beats_snapkv = (
            eligible_metrics[PRIMARY]["candidate_accuracy"]
            > eligible_metrics["snapkv_upstream_query_aware"]["candidate_accuracy"]
        )
        if unary_comparison["candidate_accuracy_family_bootstrap_95"][0] > 0 and beats_knorm and beats_snapkv:
            decision = "BEST_SCENARIO_SUPPORTED_PROCEED_G1B"
        elif beats_knorm and unary_comparison["candidate_accuracy_lift"] >= 0:
            decision = "MEDIUM_SCENARIO_PROCEED_G1B"
        elif unary_comparison["candidate_accuracy_lift"] >= 0:
            decision = "INCONCLUSIVE_PAIR_SIGNAL_PROCEED_G1B_CAUTION"
        else:
            b10 = budgets.get("10", primary_budget)
            b10_lift = b10["comparisons"][f"pair_vs_{UNARY}_full_correct"]["candidate_accuracy_lift"]
            decision = "NEGATIVE_REPLICATION_REQUIRED" if b10_lift < 0 else "MIXED_BUDGET_EFFECT"

    report = {
        "stage": "G1a",
        "decision": decision,
        "preregistered_primary_budget": 6,
        "primary_pair_method": PRIMARY,
        "unary_control": UNARY,
        "integrity_failures": failures,
        "samples": len(ids),
        "full_context_correct_samples": len(eligible),
        "budgets": budgets,
        "interpretation_scope": "Adjacent-block interaction within a shared KNorm-anchored candidate pool",
    }
    (output / "stage_g1_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage G1a budget stress", "", f"**Decision: {decision}**", "",
        "Primary endpoint: candidate accuracy at B=6 on the frozen full-context-correct cohort.", "",
        "| Budget | Pair | Unary | Block KNorm | SnapKV | Pair support | Unary support | Mask agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in args.budgets:
        value = budgets.get(str(budget))
        if value is None:
            continue
        metrics = value["cohorts"]["full_context_candidate_correct"]
        masks = value["mask_diagnostics_full_correct"]
        lines.append(
            f"| {budget} | {metrics[PRIMARY]['candidate_accuracy']:.1%} | {metrics[UNARY]['candidate_accuracy']:.1%} | "
            f"{metrics['shared_block_knorm']['candidate_accuracy']:.1%} | "
            f"{metrics['snapkv_upstream_query_aware']['candidate_accuracy']:.1%} | "
            f"{metrics[PRIMARY]['support_retention']:.1%} | {metrics[UNARY]['support_retention']:.1%} | "
            f"{masks['exact_mask_agreement']:.1%} |"
        )
    lines.extend([
        "", "G1a reuses the F2 families and therefore calibrates difficulty; it is not an independent confirmation.",
        "The pair claim is limited to adjacent boundaries in the shared candidate pool. G1b must use fresh families and include separated-support and segmentation controls.",
        "SnapKV is a matched-cache external baseline in its native query-in-prefill protocol; it is not a causal-protocol twin of BIKV.", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
