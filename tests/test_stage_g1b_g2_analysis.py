from scripts.analyze_stage_g1b_g2 import paired_family_bootstrap

import json
from pathlib import Path


def _row(family: str, value: float) -> dict:
    return {"counterfactual_group": family, "value": value}


def test_family_bootstrap_uses_families_as_independent_units() -> None:
    left = {
        "a0": _row("a", 1.0), "a1": _row("a", 1.0),
        "b0": _row("b", 0.0), "b1": _row("b", 0.0),
    }
    right = {key: _row(row["counterfactual_group"], 0.0) for key, row in left.items()}
    result = paired_family_bootstrap(left, right, set(left), lambda row: row["value"], 7)
    assert result["samples"] == 4
    assert result["families"] == 2
    assert result["mean_difference"] == 0.5


def test_primary_matched_ablation_uses_pair_alpha_and_only_unary_features() -> None:
    models = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "v14_g1b_g2_frozen_models.json").read_text()
    )
    unary = models["matched_unary_ridge_v2"]
    pair = models["pair_dp_no_query_pair_v2"]
    assert unary["kind"] == pair["kind"] == "decomposable_ridge"
    assert unary["alpha"] == pair["alpha"]
    assert all(name.startswith("unary_mean__") for name in unary["feature_names"])
    assert set(unary["feature_names"]) < set(pair["feature_names"])
