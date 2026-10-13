import pytest

from bikv.runner import necessity_passes


def test_necessity_requires_full_success_all_drops_and_all_ablation_failures():
    wrong = {"without_left": False, "without_right": False, "without_both": False}
    drops = {"without_left": 0.3, "without_right": 0.2, "without_both": 0.5}
    assert necessity_passes(True, wrong, drops, minimum_drop=0.1, require_ablation_failure=True)
    assert not necessity_passes(False, wrong, drops, minimum_drop=0.1, require_ablation_failure=True)
    assert not necessity_passes(
        True,
        {**wrong, "without_right": True},
        drops,
        minimum_drop=0.1,
        require_ablation_failure=True,
    )
    assert not necessity_passes(
        True,
        wrong,
        {**drops, "without_left": 0.05},
        minimum_drop=0.1,
        require_ablation_failure=True,
    )


def test_necessity_rejects_incomplete_ablation_sets():
    with pytest.raises(ValueError, match="all three"):
        necessity_passes(True, {}, {}, minimum_drop=0.1, require_ablation_failure=True)
