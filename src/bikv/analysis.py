from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from bikv.config import ProjectConfig
from bikv.runner import eligible_story_ids_for_variants
from bikv.runner import make_store
from bikv.schema import load_samples


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def build_report(config: ProjectConfig) -> dict[str, Any]:
    store = make_store(config)
    sample_list = load_samples(config.resolve("data_path"))
    samples = {sample.sample_id: sample for sample in sample_list}
    evaluation_variants = set(config.get("evaluation_variants", ["neutral"]))
    all_evaluation_rows: list[dict[str, Any]] = []
    evaluation_root = store.root / "evaluation"
    if evaluation_root.exists():
        for path in evaluation_root.rglob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            all_evaluation_rows.append(row)

    test_story_ids = {
        sample.story_id
        for sample in samples.values()
        if sample.split == "test" and sample.variant in evaluation_variants
    }
    necessity_story_ids = (
        eligible_story_ids_for_variants(store, sample_list, evaluation_variants) & test_story_ids
    )
    necessity_rows = []
    for sample in sample_list:
        if (
            sample.split == "test"
            and sample.variant in evaluation_variants
            and store.exists("necessity", sample.sample_id)
        ):
            necessity_rows.append(store.read_json("necessity", sample.sample_id))
    individual_passed_story_ids = {
        str(row["story_id"]) for row in necessity_rows if bool(row.get("passed"))
    }
    test_families: dict[str, set[str]] = defaultdict(set)
    for sample in sample_list:
        if sample.split == "test" and sample.variant in evaluation_variants:
            family = str(sample.metadata.get("counterfactual_group", sample.story_id))
            test_families[family].add(sample.story_id)
    passed_families = {
        family
        for family, story_ids in test_families.items()
        if story_ids and story_ids.issubset(individual_passed_story_ids)
    }
    necessity_family_pass_rate = (
        len(passed_families) / len(test_families) if test_families else 0.0
    )
    necessity_variant_pass_rate = (
        mean(float(row["passed"]) for row in necessity_rows) if necessity_rows else 0.0
    )
    symbolic_pass_rate = (
        mean(float(row.get("symbolic_passed", False)) for row in necessity_rows)
        if necessity_rows
        else 0.0
    )
    necessity_full_accuracy = (
        mean(float(row["full_generation"]["is_correct"]) for row in necessity_rows)
        if necessity_rows
        else 0.0
    )
    necessity_ablation_accuracy = {
        name: _safe_mean(
            [
                float(row["ablation_generations"][name]["is_correct"])
                for row in necessity_rows
                if name in row["ablation_generations"]
            ]
        )
        for name in ("without_left", "without_right", "without_both")
    }
    necessity_drops = {
        name: _safe_mean(
            [float(row["logprob_drops"][name]) for row in necessity_rows if name in row["logprob_drops"]]
        )
        for name in ("without_left", "without_right", "without_both")
    }
    full_by_story: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in all_evaluation_rows:
        if row["method"] == "full":
            full_by_story[row["story_id"]][row["variant"]] = bool(row["result"]["is_correct"])
    eligible_story_ids = {
        story_id
        for story_id, variants in full_by_story.items()
        if story_id in necessity_story_ids
        and set(variants) == evaluation_variants
        and all(variants.values())
    }
    surface_pass_rate = (
        len(eligible_story_ids) / len(necessity_story_ids) if necessity_story_ids else 0.0
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_evaluation_rows:
        if row["story_id"] in eligible_story_ids:
            grouped[(row["method"], row["variant"])].append(row)

    metrics: dict[str, Any] = {}
    for (method, variant), rows in sorted(grouped.items()):
        metrics.setdefault(method, {})[variant] = {
            "n": len(rows),
            "accuracy": _safe_mean([float(row["result"]["is_correct"]) for row in rows]),
            "candidate_accuracy": _safe_mean(
                [float(row["candidate_result"]["is_correct"]) for row in rows if "candidate_result" in row]
            ),
            "candidate_margin": _safe_mean(
                [float(row["candidate_result"]["gold_margin"]) for row in rows if "candidate_result" in row]
            ),
            "support_retention": _safe_mean([float(row["support_retained_final_layer"]) for row in rows]),
            "mean_seconds": _safe_mean([float(row["result"]["elapsed_seconds"]) for row in rows]),
            "mean_peak_vram_gb": _safe_mean(
                [float(row["result"]["peak_vram_bytes"]) / (1024**3) for row in rows]
            ),
        }

    order_metrics: dict[str, Any] = {}
    for row in all_evaluation_rows:
        if row["story_id"] not in eligible_story_ids:
            continue
        order = str(row.get("evidence_order", "unknown"))
        method = str(row["method"])
        bucket = order_metrics.setdefault(method, {}).setdefault(order, [])
        bucket.append(row)
    order_metrics = {
        method: {
            order: {
                "n": len(rows),
                "accuracy": _safe_mean([float(row["result"]["is_correct"]) for row in rows]),
                "candidate_accuracy": _safe_mean(
                    [float(row["candidate_result"]["is_correct"]) for row in rows if "candidate_result" in row]
                ),
                "candidate_margin": _safe_mean(
                    [float(row["candidate_result"]["gold_margin"]) for row in rows if "candidate_result" in row]
                ),
            }
            for order, rows in orders.items()
        }
        for method, orders in order_metrics.items()
    }

    label_variants = set(config.get("label_variants", ["neutral"]))
    development_story_ids = (
        eligible_story_ids_for_variants(store, sample_list, label_variants) - test_story_ids
    )
    proposal_utility: dict[str, list[float]] = defaultdict(list)
    interaction_effects: dict[str, list[float]] = defaultdict(list)
    support_interaction_by_order: dict[str, list[float]] = defaultdict(list)
    for sample_id, sample in samples.items():
        if sample.split == "test" or sample.story_id not in development_story_ids or not store.exists("labels", sample_id):
            continue
        label = store.read_json("labels", sample_id)
        for edge in label["edge_labels"]:
            kind = "support" if bool(edge["is_support_boundary"]) else "non_support"
            proposal_utility[kind].append(float(edge["proposal_utility"]))
            if "intervention_effects" in edge:
                interaction = float(edge["intervention_effects"]["interaction"])
                interaction_effects[kind].append(interaction)
                if bool(edge["is_support_boundary"]):
                    support_interaction_by_order[
                        str(label.get("evidence_order", "unknown"))
                    ].append(interaction)
    proposal_summary = {kind: _safe_mean(values) for kind, values in proposal_utility.items()}
    interaction_summary = {kind: _safe_mean(values) for kind, values in interaction_effects.items()}
    order_interaction_summary = {
        order: _safe_mean(values) for order, values in support_interaction_by_order.items()
    }

    predictor_metrics = None
    if store.exists("models", "pair_model"):
        predictor_metrics = store.read_json("models", "pair_model").get("validation_metrics")

    def accuracy(method: str, variant: str) -> float | None:
        return metrics.get(method, {}).get(variant, {}).get("accuracy")

    minimum_eligible_test_stories = int(config.get("necessity_min_eligible_test_stories", 1))
    necessity_signal = (
        necessity_family_pass_rate >= float(config.get("necessity_min_family_pass_rate", 0.6))
        and len(necessity_story_ids) >= minimum_eligible_test_stories
    )
    full_stable = bool(necessity_story_ids) and surface_pass_rate >= 0.8
    primary_variant = sorted(evaluation_variants)[0]
    singleton_primary = accuracy("singleton", primary_variant)
    oracle_primary = accuracy("oracle", primary_variant)
    learned_primary = accuracy("learned", primary_variant)
    neighbor_primary = accuracy("neighbor", primary_variant)
    oracle_gap = None
    learned_recovery = None
    neighbor_recovery = None
    if singleton_primary is not None and oracle_primary is not None:
        oracle_gap = oracle_primary - singleton_primary
        if oracle_gap > 1e-9 and learned_primary is not None:
            learned_recovery = (learned_primary - singleton_primary) / oracle_gap
        if oracle_gap > 1e-9 and neighbor_primary is not None:
            neighbor_recovery = (neighbor_primary - singleton_primary) / oracle_gap

    validation_gate = bool(
        store.exists("models", "pair_model")
        and store.read_json("models", "pair_model").get("validation_gate_passed")
    )
    predictor_signal = validation_gate
    method_signal = bool(
        learned_primary is not None
        and singleton_primary is not None
        and (
            learned_primary - singleton_primary >= 0.05
            or (learned_recovery is not None and learned_recovery >= 0.3)
        )
    )
    neighbor_not_enough = neighbor_recovery is None or neighbor_recovery < 0.8

    if necessity_signal and full_stable and predictor_signal and method_signal and neighbor_not_enough:
        decision = "GO"
    elif necessity_signal and full_stable and predictor_signal:
        decision = "REFINE"
    else:
        decision = "STOP_OR_REDESIGN"

    report = {
        "config_hash": config.config_hash,
        "metrics": metrics,
        "metrics_by_evidence_order": order_metrics,
        "proposal_utility": proposal_summary,
        "interaction_effect": interaction_summary,
        "support_interaction_by_evidence_order": order_interaction_summary,
        "label_semantics": "exact_budget_2x2_difference_in_differences",
        "necessity": {
            "family_pass_rate": necessity_family_pass_rate,
            "variant_pass_rate": necessity_variant_pass_rate,
            "individual_passed_stories": len(individual_passed_story_ids),
            "individual_total_stories": len(test_story_ids),
            "complete_passed_families": len(passed_families),
            "total_families": len(test_families),
            "family_eligible_stories": len(necessity_story_ids),
            "symbolic_pass_rate": symbolic_pass_rate,
            "full_accuracy": necessity_full_accuracy,
            "ablation_accuracy": necessity_ablation_accuracy,
            "mean_logprob_drop": necessity_drops,
        },
        "predictor_metrics": predictor_metrics,
        "gates": {
            "full_context_stable": full_stable,
            "necessity_signal": necessity_signal,
            "necessity_family_pass_rate": necessity_family_pass_rate,
            "necessity_variant_pass_rate": necessity_variant_pass_rate,
            "symbolic_pass_rate": symbolic_pass_rate,
            "necessity_eligible_test_stories": len(necessity_story_ids),
            "minimum_eligible_test_stories": minimum_eligible_test_stories,
            "surface_filter_pass_rate": surface_pass_rate,
            "eligible_test_stories": len(eligible_story_ids),
            "total_test_stories": len(test_story_ids),
            "validation_gate": validation_gate,
            "predictor_signal": predictor_signal,
            "method_signal": method_signal,
            "neighbor_not_enough": neighbor_not_enough,
            "oracle_gap": oracle_gap,
            "learned_recovery": learned_recovery,
            "neighbor_recovery": neighbor_recovery,
        },
        "decision": decision,
    }
    store.write_json("reports", "decision", report)
    markdown = _render_markdown(report)
    store.write_text("reports", "pilot_report", markdown)
    _write_plots(report, store.root / "reports")
    store.bundle()
    return report


def _write_plots(report: dict[str, Any], output_dir) -> None:
    if not report["metrics"]:
        return
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    variants = sorted({variant for rows in report["metrics"].values() for variant in rows})
    methods = list(report["metrics"])
    width = 0.8 / max(1, len(methods))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for method_index, method in enumerate(methods):
        values = [report["metrics"].get(method, {}).get(variant, {}).get("accuracy", 0.0) or 0.0 for variant in variants]
        positions = [index - 0.4 + width / 2 + method_index * width for index in range(len(variants))]
        axis.bar(positions, values, width=width, label=method)
    axis.set_xticks(range(len(variants)), variants)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Exact-match accuracy")
    axis.set_title("Matched-budget evaluation after the full-context filter")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_by_variant.png", dpi=160)
    plt.close(figure)

    proposal_utility = report["proposal_utility"]
    if proposal_utility:
        names = [name for name in ("support", "non_support") if proposal_utility.get(name) is not None]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(names, [proposal_utility[name] for name in names], color=["#4666A5", "#4E8A62"][: len(names)])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("Mean candidate margin")
        axis.set_title("Exact-budget pair proposal utility")
        figure.tight_layout()
        figure.savefig(output_dir / "proposal_utility.png", dpi=160)
        plt.close(figure)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Boundary-Interaction KV pilot report",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Evaluation",
        "",
        "| Method | Variant | N | Greedy acc. | Candidate acc. | Margin | Support retained | Seconds | Peak VRAM GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, variants in report["metrics"].items():
        for variant, values in variants.items():
            lines.append(
                f"| {method} | {variant} | {values['n']} | {_format(values['accuracy'])} | "
                f"{_format(values['candidate_accuracy'])} | {_format(values['candidate_margin'])} | "
                f"{_format(values['support_retention'])} | {_format(values['mean_seconds'])} | "
                f"{_format(values['mean_peak_vram_gb'])} |"
            )
    lines.extend(["", "## Evidence-order control", ""])
    for method, orders in report.get("metrics_by_evidence_order", {}).items():
        for order, values in orders.items():
            lines.append(
                f"- `{method}/{order}`: n={values['n']}, greedy={_format(values['accuracy'])}, "
                f"candidate={_format(values['candidate_accuracy'])}, margin={_format(values['candidate_margin'])}"
            )
    lines.extend(["", "## Gates", ""])
    for name, value in report["gates"].items():
        lines.append(f"- `{name}`: {_format(value)}")
    lines.extend(["", "## Causal necessity", ""])
    lines.append(f"- `family_pass_rate`: {_format(report['necessity']['family_pass_rate'])}")
    lines.append(f"- `variant_pass_rate`: {_format(report['necessity']['variant_pass_rate'])}")
    lines.append(
        "- `individual_passed_stories`: "
        f"{report['necessity']['individual_passed_stories']}/"
        f"{report['necessity']['individual_total_stories']}"
    )
    lines.append(
        "- `complete_passed_families`: "
        f"{report['necessity']['complete_passed_families']}/"
        f"{report['necessity']['total_families']}"
    )
    lines.append(f"- `family_eligible_stories`: {report['necessity']['family_eligible_stories']}")
    lines.append(f"- `symbolic_pass_rate`: {_format(report['necessity']['symbolic_pass_rate'])}")
    lines.append(f"- `full_accuracy`: {_format(report['necessity']['full_accuracy'])}")
    for name, value in report["necessity"]["ablation_accuracy"].items():
        lines.append(f"- `{name}_accuracy`: {_format(value)}")
    for name, value in report["necessity"]["mean_logprob_drop"].items():
        lines.append(f"- `{name}_mean_logprob_drop`: {_format(value)}")
    lines.extend(["", "## Exact-budget proposal utility", ""])
    for kind, value in report["proposal_utility"].items():
        lines.append(f"- `{kind}`: {_format(value)}")
    lines.extend(["", "## Matched 2x2 interaction effect", ""])
    for kind, value in report.get("interaction_effect", {}).items():
        lines.append(f"- `{kind}`: {_format(value)}")
    for order, value in report.get("support_interaction_by_evidence_order", {}).items():
        lines.append(f"- `support/{order}`: {_format(value)}")
    if report["predictor_metrics"]:
        lines.extend(["", "## Predictor", ""])
        for name, value in report["predictor_metrics"].items():
            lines.append(f"- `{name}`: {_format(value)}")
    return "\n".join(lines) + "\n"
