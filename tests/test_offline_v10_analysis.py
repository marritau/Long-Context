from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "offline_v10_analysis.py"
SPEC = importlib.util.spec_from_file_location("offline_v10_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(gold: float, wrong: list[float]) -> dict:
    scores = {"GOLD": gold, **{f"W{index}": value for index, value in enumerate(wrong)}}
    return {"scores": scores, "gold_score": gold}


def test_candidate_log_odds_uses_all_competitors() -> None:
    utilities = MODULE.result_utilities(_result(-1.0, [-2.0, -3.0]), "GOLD")
    assert utilities["max_margin"] == pytest.approx(1.0)
    assert utilities["gold_logprob"] == pytest.approx(-1.0)
    assert utilities["candidate_log_odds"] < utilities["max_margin"]


def test_gold_identity_is_not_inferred_from_a_tied_score() -> None:
    result = {"scores": {"GOLD": -1.0, "TIED_WRONG": -1.0, "OTHER": -3.0}, "gold_score": -1.0}
    utilities = MODULE.result_utilities(result, "GOLD")
    assert utilities["max_margin"] == pytest.approx(0.0)
    assert MODULE.wrong_winner(result, "GOLD") == "TIED_WRONG"


def test_factorial_interaction_cancels_additive_effects() -> None:
    cell = {
        "both": 8.0,
        "left_only": 5.0,
        "right_only": 6.0,
        "neither": 3.0,
    }
    interaction = cell["both"] - cell["left_only"] - cell["right_only"] + cell["neither"]
    assert interaction == pytest.approx(0.0)
