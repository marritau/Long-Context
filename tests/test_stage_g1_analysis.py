from scripts.analyze_stage_g1 import mask_diagnostics, paired_family_bootstrap


def _row(family: str, correct: bool, margin: float, mask: list[int]) -> dict:
    return {
        "counterfactual_group": family,
        "candidate_result": {"is_correct": correct, "gold_margin": margin},
        "selected_blocks": mask,
        "supporting_blocks": [0, 1],
    }


def test_family_bootstrap_and_mcnemar_use_paired_outcomes():
    ids = ["a", "b", "c", "d"]
    proposed = {
        sample_id: _row("f1" if sample_id in {"a", "b"} else "f2", True, 2.0, [0, 1])
        for sample_id in ids
    }
    baseline = {
        sample_id: _row("f1" if sample_id in {"a", "b"} else "f2", False, 0.0, [1, 2])
        for sample_id in ids
    }
    result = paired_family_bootstrap(
        {"proposed": proposed, "baseline": baseline}, ids, "proposed", "baseline", draws=100
    )
    assert result["candidate_accuracy_lift"] == 1.0
    assert result["candidate_accuracy_family_bootstrap_95"] == [1.0, 1.0]
    assert result["proposed_only_correct"] == 4
    assert result["baseline_only_correct"] == 0
    assert result["mcnemar_exact_two_sided_p"] == 0.125


def test_mask_diagnostics_detect_pair_only_support_retention():
    results = {
        "candidate_pair_no_query_v2": {"a": _row("f1", True, 1.0, [0, 1])},
        "candidate_query_unary_v2": {"a": _row("f1", False, -1.0, [1, 2])},
    }
    diagnostics = mask_diagnostics(results, ["a"])
    assert diagnostics["exact_mask_agreement"] == 0.0
    assert diagnostics["mean_mask_jaccard"] == 1 / 3
    assert diagnostics["pair_only_retains_support"] == 1
    assert diagnostics["unary_only_retains_support"] == 0
