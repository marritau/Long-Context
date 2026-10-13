from scripts.offline_stage_h import build_graph_actions


def test_graph_action_builder_uses_same_target_and_matched_unary_subset() -> None:
    rows = [
        {
            "sample_id": "s",
            "proposal_mask": [0, 2],
            "counterfactual_group": "f",
            "split": "train",
            "evidence_order": "forward",
            "supporting_blocks": [0, 2],
            "proposal_utilities": {"candidate_log_odds": 1.5},
            "is_support_boundary": False,
        }
    ]
    captures = {
        "s": {
            "legacy_unary": [0.0, 0.0, 0.0],
            "blocks": [
                {"query_attention_head_mean__all": value, "knorm_head_mean__all": value}
                for value in (0.2, 0.1, 0.3)
            ],
        }
    }
    sample = type("Sample", (), {"blocks": ["node A", "noise", "A gives code"]})()
    actions, all_names, unary_names = build_graph_actions(rows, captures, {"s": sample})
    assert actions[0]["target"] == 1.5
    assert set(unary_names) < set(all_names)
    assert all(name.startswith("unary_mean__") for name in unary_names)
    assert not any("distance_fraction" in name or "is_adjacent" in name for name in all_names)
