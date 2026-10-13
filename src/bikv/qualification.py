from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _rate(rows: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values: list[bool] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(bool(value))
    successes = sum(values)
    return {
        "successes": successes,
        "total": len(values),
        "rate": successes / len(values) if values else None,
        "wilson_95": wilson_interval(successes, len(values)),
    }


def summarize_qualification(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    neutral = [row for row in rows if row["variant"] == "neutral"]
    by_variant = {
        variant: _rate([row for row in rows if row["variant"] == variant], ("full_candidates", "is_correct"))
        for variant in ("neutral", "targeted", "sham")
    }
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in neutral:
        by_position[str(row["support_position_bucket"])].append(row)

    story_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        story_groups[str(row["story_id"])].append(row)
        family_groups[str(row["counterfactual_group"])].append(row)
    stable_stories = [
        all(bool(item["full_candidates"]["is_correct"]) for item in group) and len(group) == 3
        for group in story_groups.values()
    ]
    greedy_stable_stories = [
        all(bool(item["full_greedy"]["is_correct"]) for item in group) and len(group) == 3
        for group in story_groups.values()
    ]
    counterfactual_pairs = []
    for group in family_groups.values():
        neutral_group = [item for item in group if item["variant"] == "neutral"]
        counterfactual_pairs.append(
            len(neutral_group) == 2
            and all(bool(item["full_candidates"]["is_correct"]) for item in neutral_group)
            and all(bool(item["chain_end_to_end"]["is_correct"]) for item in neutral_group)
        )

    ablation_names = ("without_left", "without_right", "without_both")
    ablation_failure = {
        name: {
            "successes": sum(not bool(row["ablations"][name]["is_correct"]) for row in neutral),
            "total": len(neutral),
        }
        for name in ablation_names
    }
    for value in ablation_failure.values():
        value["rate"] = value["successes"] / value["total"] if value["total"] else None
        value["wilson_95"] = wilson_interval(value["successes"], value["total"])
    necessity_values = [
        bool(row["full_candidates"]["is_correct"])
        and all(not bool(row["ablations"][name]["is_correct"]) for name in ablation_names)
        for row in neutral
    ]

    metrics = {
        "full_candidate": _rate(rows, ("full_candidates", "is_correct")),
        "neutral_full_candidate": _rate(neutral, ("full_candidates", "is_correct")),
        "full_greedy": _rate(rows, ("full_greedy", "is_correct")),
        "by_variant": by_variant,
        "by_support_position": {
            bucket: _rate(group, ("full_candidates", "is_correct")) for bucket, group in sorted(by_position.items())
        },
        "hop1_agent": _rate(neutral, ("hop1_agent", "is_correct")),
        "hop2_code_given_gold_agent": _rate(neutral, ("hop2_code_given_gold_agent", "is_correct")),
        "chain_end_to_end": _rate(neutral, ("chain_end_to_end", "is_correct")),
        "story_variant_stability": {
            "successes": sum(stable_stories),
            "total": len(stable_stories),
            "rate": _mean([float(value) for value in stable_stories]),
            "wilson_95": wilson_interval(sum(stable_stories), len(stable_stories)),
        },
        "greedy_story_variant_stability": {
            "successes": sum(greedy_stable_stories),
            "total": len(greedy_stable_stories),
            "rate": _mean([float(value) for value in greedy_stable_stories]),
            "wilson_95": wilson_interval(sum(greedy_stable_stories), len(greedy_stable_stories)),
        },
        "counterfactual_pair_pass": {
            "successes": sum(counterfactual_pairs),
            "total": len(counterfactual_pairs),
            "rate": _mean([float(value) for value in counterfactual_pairs]),
            "wilson_95": wilson_interval(sum(counterfactual_pairs), len(counterfactual_pairs)),
        },
        "neutral_candidate_necessity": {
            "successes": sum(necessity_values),
            "total": len(necessity_values),
            "rate": _mean([float(value) for value in necessity_values]),
            "wilson_95": wilson_interval(sum(necessity_values), len(necessity_values)),
        },
        "ablation_failure": ablation_failure,
        "mean_full_gold_margin": _mean([float(row["full_candidates"]["gold_margin"]) for row in rows]),
        "mean_ablation_margin_drop": {
            name: _mean(
                [
                    float(row["full_candidates"]["gold_margin"])
                    - float(row["ablations"][name]["gold_margin"])
                    for row in neutral
                ]
            )
            for name in ablation_names
        },
    }

    checks = {
        # The expensive labels use neutral samples only. Surface variants have
        # their own robustness gate and must not silently redefine task solvability.
        "neutral_candidate_accuracy": metrics["neutral_full_candidate"]["rate"]
        >= thresholds["min_candidate_accuracy"],
        "full_greedy_accuracy": metrics["full_greedy"]["rate"] >= thresholds["min_greedy_accuracy"],
        "worst_variant_accuracy": min(value["rate"] for value in by_variant.values()) >= thresholds["min_variant_accuracy"],
        "hop1_accuracy": metrics["hop1_agent"]["rate"] >= thresholds["min_hop_accuracy"],
        "hop2_accuracy": metrics["hop2_code_given_gold_agent"]["rate"] >= thresholds["min_hop_accuracy"],
        "chain_accuracy": metrics["chain_end_to_end"]["rate"] >= thresholds["min_chain_accuracy"],
        "story_variant_stability": metrics["story_variant_stability"]["rate"] >= thresholds["min_story_stability"],
        "greedy_story_variant_stability": metrics["greedy_story_variant_stability"]["rate"]
        >= thresholds["min_greedy_story_stability"],
        "counterfactual_pair_pass": metrics["counterfactual_pair_pass"]["rate"] >= thresholds["min_counterfactual_pair_rate"],
        "candidate_necessity": metrics["neutral_candidate_necessity"]["rate"] >= thresholds["min_necessity_rate"],
        "support_position_robustness": min(
            value["rate"] for value in metrics["by_support_position"].values()
        )
        >= thresholds["min_position_accuracy"],
    }
    return {
        "decision": "GO_LABEL" if all(checks.values()) else "REFINE_BACKBONE",
        "sample_count": len(rows),
        "neutral_story_count": len(neutral),
        "thresholds": thresholds,
        "checks": checks,
        "metrics": metrics,
    }


def render_qualification_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    lines = [
        "# Backbone qualification report",
        "",
        f"**Decision: {summary['decision']}**",
        "",
        f"Samples: {summary['sample_count']}; neutral latent stories: {summary['neutral_story_count']}.",
        "",
        "## Core metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Candidate-ranked full-context accuracy | {pct(metrics['full_candidate']['rate'])} |",
        f"| Neutral candidate accuracy (label population) | {pct(metrics['neutral_full_candidate']['rate'])} |",
        f"| Greedy full-context accuracy | {pct(metrics['full_greedy']['rate'])} |",
        f"| Hop 1: laboratory to technician | {pct(metrics['hop1_agent']['rate'])} |",
        f"| Hop 2: technician to code, gold technician supplied | {pct(metrics['hop2_code_given_gold_agent']['rate'])} |",
        f"| End-to-end decomposed chain | {pct(metrics['chain_end_to_end']['rate'])} |",
        f"| Stable across neutral/targeted/sham | {pct(metrics['story_variant_stability']['rate'])} |",
        f"| Greedy-stable across neutral/targeted/sham | {pct(metrics['greedy_story_variant_stability']['rate'])} |",
        f"| Counterfactual pair pass | {pct(metrics['counterfactual_pair_pass']['rate'])} |",
        f"| Strict pre-prefill text necessity on neutral stories | {pct(metrics['neutral_candidate_necessity']['rate'])} |",
        "",
        "## Variant and position controls",
        "",
        "| Group | Candidate accuracy |",
        "|---|---:|",
    ]
    for variant, value in metrics["by_variant"].items():
        lines.append(f"| variant={variant} | {pct(value['rate'])} |")
    for bucket, value in metrics["by_support_position"].items():
        lines.append(f"| support_position={bucket} | {pct(value['rate'])} |")
    lines.extend(["", "## Gate checks", ""])
    for name, passed in summary["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "`GO_LABEL` means the backbone is strong enough to start expensive KV interventions. "
            "It is not evidence that the proposed compression method works.",
            "",
        ]
    )
    return "\n".join(lines)
