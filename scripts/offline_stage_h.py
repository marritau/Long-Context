from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from bikv.graph import graph_action_features
from bikv.schema import load_samples


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from offline_stage_e import (
        freeze_dp,
        freeze_dp_at_alpha,
        metrics_by_order,
        nested_dp,
        paired_family_bootstrap,
    )
    from offline_stage_e2 import load_training_captures
    from offline_v10_analysis import build_rows
except ModuleNotFoundError:
    from scripts.offline_stage_e import (
        freeze_dp,
        freeze_dp_at_alpha,
        metrics_by_order,
        nested_dp,
        paired_family_bootstrap,
    )
    from scripts.offline_stage_e2 import load_training_captures
    from scripts.offline_v10_analysis import build_rows


TARGET_KEY = "candidate_log_odds"


def build_graph_actions(
    boundary_rows: list[dict[str, Any]],
    captures: dict[str, dict[str, Any]],
    samples: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        documents[str(row["sample_id"])].append(row)
    actions: list[dict[str, Any]] = []
    all_names: list[str] | None = None
    unary_names: list[str] | None = None
    for sample_id, rows in documents.items():
        if sample_id not in captures or sample_id not in samples:
            continue
        aliases: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            aliases[tuple(row["proposal_mask"])].append(row)
        for mask, members in aliases.items():
            features = graph_action_features(
                list(mask), captures[sample_id], samples[sample_id].blocks, include_distance=False
            )
            names = sorted(features)
            current_unary = [name for name in names if name.startswith("unary_mean__")]
            all_names = names if all_names is None else all_names
            unary_names = current_unary if unary_names is None else unary_names
            if names != all_names or current_unary != unary_names:
                raise ValueError("Stage H graph feature schema changed between actions")
            support = set(members[0]["supporting_blocks"])
            actions.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_group": members[0]["counterfactual_group"],
                    "split": members[0]["split"],
                    "evidence_order": members[0]["evidence_order"],
                    "mask": list(mask),
                    "features": features,
                    "dp_features": features,
                    "target": float(members[0]["proposal_utilities"][TARGET_KEY]),
                    "retains_support": support.issubset(mask),
                    "same_as_support_proposal": any(member["is_support_boundary"] for member in members),
                }
            )
    if not actions or all_names is None or unary_names is None:
        raise ValueError("No Stage H graph actions could be constructed")
    return actions, all_names, unary_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage H CPU audit for arbitrary-pair graph scoring")
    parser.add_argument("v10_run", type=Path)
    parser.add_argument("rich_run", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    v10 = args.v10_run.resolve()
    output = (args.output or v10 / "offline_stage_h").resolve()
    output.mkdir(parents=True, exist_ok=True)

    boundary_rows, manifest = build_rows(v10, args.data.resolve())
    captures = load_training_captures(args.rich_run.resolve(), v10)
    samples = {sample.sample_id: sample for sample in load_samples(args.data.resolve())}
    actions, graph_names, unary_names = build_graph_actions(boundary_rows, captures, samples)

    graph_scores, graph_alphas = nested_dp(actions, graph_names)
    unary_scores, unary_alphas = nested_dp(actions, unary_names)
    graph_result = metrics_by_order(actions, graph_scores)
    unary_result = metrics_by_order(actions, unary_scores)
    comparison = paired_family_bootstrap(actions, graph_scores, unary_scores)

    graph_model = freeze_dp(actions, graph_names)
    unary_model = freeze_dp_at_alpha(actions, unary_names, float(graph_model["alpha"]))
    frozen = {
        "graph_matched_unary_ridge_v1": unary_model,
        "graph_arbitrary_pair_ridge_v1": graph_model,
    }
    report = {
        "stage": "H_CPU",
        "scope": "arbitrary pairs inside a query-aware shortlist; no distance feature in primary model",
        "cohort": manifest["offline_cohort"],
        "actions": len(actions),
        "families": len({row["counterfactual_group"] for row in actions}),
        "feature_counts": {"matched_unary": len(unary_names), "arbitrary_pair": len(graph_names)},
        "nested_selected_alphas": {"unary": unary_alphas, "pair": graph_alphas},
        "results": {"matched_unary": unary_result, "arbitrary_pair": graph_result},
        "paired_pair_vs_unary": comparison,
        "frozen_models": frozen,
    }
    (output / "stage_h_cpu.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "frozen_models.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Stage H CPU arbitrary-pair audit",
        "",
        "This is a grouped offline audit on existing V10 actions, not an independent GPU result.",
        "Distance and adjacency indicators are excluded from the primary graph model.",
        "",
        "| Model | Support retention | Utility regret |",
        "|---|---:|---:|",
        f"| Matched unary Ridge | {unary_result['all']['action_support_retention']:.1%} | {unary_result['all']['mean_action_utility_regret']:.3f} |",
        f"| Arbitrary-pair Ridge | {graph_result['all']['action_support_retention']:.1%} | {graph_result['all']['mean_action_utility_regret']:.3f} |",
        "",
        f"Pair-minus-unary retention: {comparison['retention_lift']:+.1%}; "
        f"family-bootstrap 95% CI {comparison['retention_lift_family_bootstrap_95']}.",
        f"Utility-regret reduction: {comparison['regret_reduction']:+.3f}; "
        f"family-bootstrap 95% CI {comparison['regret_reduction_family_bootstrap_95']}.",
        "",
        "The GPU confirmation must use fresh chain families and the frozen models above.",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
