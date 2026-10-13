from bikv.qualification import summarize_qualification, wilson_interval


def _candidate(correct=True, margin=1.0):
    return {"is_correct": correct, "gold_margin": margin}


def _row(story, variant, family, position, correct=True):
    row = {
        "story_id": story,
        "variant": variant,
        "counterfactual_group": family,
        "support_position_bucket": position,
        "full_candidates": _candidate(correct),
        "full_greedy": {"is_correct": correct},
    }
    if variant == "neutral":
        row.update(
            {
                "hop1_agent": {"is_correct": correct},
                "hop2_code_given_gold_agent": {"is_correct": correct},
                "chain_end_to_end": {"is_correct": correct},
                "ablations": {
                    "without_left": _candidate(False, -0.5),
                    "without_right": _candidate(False, -0.4),
                    "without_both": _candidate(False, -0.8),
                },
            }
        )
    return row


def test_qualification_summary_requires_reasoning_controls():
    rows = []
    positions = ["early", "middle", "late"]
    for index in range(6):
        for variant in ("neutral", "targeted", "sham"):
            rows.append(_row(f"story-{index}", variant, f"family-{index // 2}", positions[index % 3]))
    thresholds = {
        "min_candidate_accuracy": 0.9,
        "min_greedy_accuracy": 0.75,
        "min_variant_accuracy": 0.8,
        "min_hop_accuracy": 0.9,
        "min_chain_accuracy": 0.85,
        "min_story_stability": 0.75,
        "min_greedy_story_stability": 0.60,
        "min_counterfactual_pair_rate": 0.75,
        "min_necessity_rate": 0.3,
        "min_position_accuracy": 0.70,
    }
    summary = summarize_qualification(rows, thresholds)
    assert summary["decision"] == "GO_LABEL"
    assert summary["metrics"]["counterfactual_pair_pass"]["rate"] == 1.0
    rows[0]["hop1_agent"]["is_correct"] = False
    rows[3]["hop1_agent"]["is_correct"] = False
    summary = summarize_qualification(rows, thresholds)
    assert summary["decision"] == "REFINE_BACKBONE"
    assert summary["checks"]["hop1_accuracy"] is False


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0
