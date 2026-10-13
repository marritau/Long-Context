from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from offline_stage_e import (
    action_features_v2,
    freeze_dp,
    freeze_dp_at_alpha,
    freeze_ranknet,
    metrics_by_order,
    nested_dp,
    nested_ranknet,
    paired_family_bootstrap,
    summarize_capture_v2,
)
from offline_v10_analysis import build_rows


TARGET_KEY = "candidate_log_odds"


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_captures(rich_root: Path, legacy_root: Path) -> dict[str, dict[str, Any]]:
    legacy = {
        path.name.removesuffix(".json.gz"): load_json_gz(path)["layers"]
        for path in (legacy_root / "capture").glob("*.json.gz")
    }
    output = {}
    for path in sorted((rich_root / "rich_capture").glob("*.json.gz")):
        payload = load_json_gz(path)
        sample_id = str(payload["sample_id"])
        output[sample_id] = summarize_capture_v2(payload, legacy_layers=legacy.get(sample_id))
    return output


def build_actions(
    boundary_rows: list[dict[str, Any]], captures: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        documents[row["sample_id"]].append(row)
    actions = []
    rich_names = None
    dp_names = None
    for sample_id, rows in documents.items():
        aliases: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            aliases[tuple(row["proposal_mask"])].append(row)
        for mask_tuple, members in aliases.items():
            rich, decomposable = action_features_v2(list(mask_tuple), captures[sample_id])
            rich_names = sorted(rich) if rich_names is None else rich_names
            dp_names = sorted(decomposable) if dp_names is None else dp_names
            if rich_names != sorted(rich) or dp_names != sorted(decomposable):
                raise ValueError("Stage E2 feature schema changed between samples")
            support = set(members[0]["supporting_blocks"])
            actions.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_group": members[0]["counterfactual_group"],
                    "split": members[0]["split"],
                    "evidence_order": members[0]["evidence_order"],
                    "mask": list(mask_tuple),
                    "features": rich,
                    "dp_features": decomposable,
                    "target": float(members[0]["proposal_utilities"][TARGET_KEY]),
                    "retains_support": support.issubset(mask_tuple),
                    "same_as_support_proposal": any(member["is_support_boundary"] for member in members),
                }
            )
    if rich_names is None or dp_names is None:
        raise ValueError("No Stage E2 actions")
    return actions, rich_names, dp_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage E2: corrected transferable pair-policy audit")
    parser.add_argument("v10_run", type=Path)
    parser.add_argument("rich_run", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.v10_run / "offline_stage_e2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_rows(args.v10_run.resolve(), args.data.resolve())
    captures = load_training_captures(args.rich_run.resolve(), args.v10_run.resolve())
    actions, rich_names, dp_names = build_actions(rows, captures)

    query_rank_names = [name for name in rich_names if "query_attention_" in name]
    unary_names = [name for name in dp_names if name.startswith("unary_mean__")]
    pair_names = [name for name in dp_names if name.startswith("pair_budget_mean__")]
    no_direction = [
        name for name in dp_names
        if "later_to_earlier" not in name and "direction_log_lift" not in name
    ]
    query_pair = [
        name for name in dp_names
        if name.startswith("unary_mean__") and "query_attention_" in name
        or name.startswith("pair_budget_mean__query_pair_")
    ]
    no_query_pair = [name for name in no_direction if "query_pair_" not in name]
    feature_sets = {
        "query_unary_ranknet": ("ranknet", query_rank_names),
        "dp_normalized_full": ("dp", dp_names),
        "dp_normalized_unary_only": ("dp", unary_names),
        "dp_normalized_pair_only": ("dp", pair_names),
        "dp_pair_aware_no_direction": ("dp", no_direction),
        "dp_query_pair_compact": ("dp", query_pair),
        "dp_no_query_pair": ("dp", no_query_pair),
    }
    scores = {}
    for name, (kind, names) in feature_sets.items():
        print(f"[stage-e2] {name}: {len(names)} features", flush=True)
        scores[name] = nested_ranknet(actions, names)[0] if kind == "ranknet" else nested_dp(actions, names)[0]
    results = {name: metrics_by_order(actions, values) for name, values in scores.items()}
    baseline = scores["query_unary_ranknet"]
    comparisons = {
        name: paired_family_bootstrap(actions, values, baseline)
        for name, values in scores.items()
        if name != "query_unary_ranknet"
    }
    # This is the frozen pair policy used prospectively from Stage F2 onward.
    # "no_query_pair" excludes explicit query-conditioned pair terms, while
    # retaining query-aware unary features.
    selected_name = "dp_no_query_pair"
    selected_ci = comparisons[selected_name]["regret_reduction_family_bootstrap_95"]
    selected_retention_ci = comparisons[selected_name]["retention_lift_family_bootstrap_95"]
    decision = "PASS_TO_F2_GPU" if (
        results[selected_name]["all"]["mean_action_utility_regret"]
        <= results["query_unary_ranknet"]["all"]["mean_action_utility_regret"]
        and (selected_ci[1] > 0 or selected_retention_ci[1] > 0)
    ) else "KEEP_UNARY_AS_PRIMARY"
    frozen_pair = freeze_dp(actions, no_query_pair)
    frozen = {
        "query_unary_ranknet_v2": freeze_ranknet(actions, query_rank_names),
        # Primary matched ablation: identical actions, target, scaling, learner
        # family and the alpha selected for the parent pair model.
        "matched_unary_ridge_v2": freeze_dp_at_alpha(
            actions, unary_names, float(frozen_pair["alpha"])
        ),
        # Secondary diagnostic showing what unary Ridge selects under its own CV.
        "matched_unary_ridge_cv_v2": freeze_dp(actions, unary_names),
        "pair_dp_compact_v2": freeze_dp(actions, query_pair),
        "pair_dp_broad_v2": freeze_dp(actions, no_direction),
        "pair_dp_no_query_pair_v2": frozen_pair,
    }
    report = {
        "stage": "E2",
        "decision": decision,
        "selected_pair_hypothesis": selected_name,
        "scientific_corrections": {
            "candidate_generator": "exact v10 layer-zscored KNorm unary",
            "unary_scaling": "mean over selected blocks",
            "pair_scaling": "sum over retained adjacent edges divided by budget-1",
            "query_pair_terms": ["geometric mean", "minimum", "sum", "balance"],
            "directed_attention": "excluded from preregistered compact policy after F1 ablation",
            "boundary_surprise_name": "token surprisal; not EM-LLM Bayesian surprise",
        },
        "cohort": manifest["offline_cohort"],
        "actions": len(actions),
        "families": len({row["counterfactual_group"] for row in actions}),
        "feature_counts": {name: len(names) for name, (_, names) in feature_sets.items()},
        "results": results,
        "paired_vs_query_unary": comparisons,
        "frozen_models": frozen,
    }
    (output / "stage_e2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "frozen_models.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage E2: corrected transferable pair-policy audit", "", f"**Decision: {decision}**", "",
        "| Method | Support retention | MRR | Utility regret |", "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        value = result["all"]
        lines.append(
            f"| {name} | {value['action_support_retention']:.1%} | {value['support_action_mrr']:.3f} | {value['mean_action_utility_regret']:.3f} |"
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
