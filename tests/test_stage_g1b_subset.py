from types import SimpleNamespace

import pytest

from bikv.g1b_data import select_g1b_family_subset


def _samples(families: int = 48):
    rows = []
    for family in range(families):
        condition = "separated" if family % 6 == 0 else "adjacent"
        for member in range(2):
            rows.append(
                SimpleNamespace(
                    sample_id=f"story-{family:02d}-{member}",
                    supporting_blocks=[family, family + (2 if condition == "separated" else 1)],
                    metadata={
                        "counterfactual_group": f"family-{family:02d}",
                        "evidence_condition": condition,
                    },
                )
            )
    return rows


def test_subset_preserves_complete_families_conditions_and_position_span():
    selected = select_g1b_family_subset(_samples(), 64)
    families = {row.metadata["counterfactual_group"] for row in selected}
    assert len(selected) == 64
    assert len(families) == 32
    assert all(sum(row.metadata["counterfactual_group"] == family for row in selected) == 2 for family in families)
    assert {row.metadata["evidence_condition"] for row in selected} == {"adjacent", "separated"}
    positions = [min(row.supporting_blocks) for row in selected]
    assert min(positions) == 0
    assert max(positions) == 47


@pytest.mark.parametrize("limit", [0, 63, 98])
def test_subset_rejects_invalid_story_limits(limit):
    with pytest.raises(ValueError):
        select_g1b_family_subset(_samples(), limit)
