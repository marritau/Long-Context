import random

from bikv.presses import matched_interaction_masks
from bikv.selector import brute_force_chain_budget, select_chain_budget, select_chain_with_cut_penalty


def test_dynamic_program_matches_brute_force():
    rng = random.Random(17)
    for n in range(1, 9):
        for _ in range(20):
            unary = [rng.uniform(-2, 3) for _ in range(n)]
            edges = [rng.uniform(-1, 2) for _ in range(max(0, n - 1))]
            costs = [rng.randint(1, 3) for _ in range(n)]
            budget = rng.randint(0, sum(costs))
            exact = select_chain_budget(unary, edges, costs, budget)
            brute = brute_force_chain_budget(unary, edges, costs, budget)
            assert abs(exact.objective - brute.objective) < 1e-8
            assert exact.used_budget <= budget


def test_positive_edge_can_preserve_a_pair():
    without_pair = select_chain_budget([2.0, 0.1, 1.9], [0.0, 0.0], [1, 1, 1], 2, exact_budget=True)
    with_pair = select_chain_budget([2.0, 0.1, 1.9], [3.0, 0.0], [1, 1, 1], 2, exact_budget=True)
    assert without_pair.selected == [0, 2]
    assert with_pair.selected == [0, 1]


def test_exact_budget_keeps_the_same_number_of_negative_items():
    result = select_chain_budget([-1.0, -2.0, -3.0], [0.0, 0.0], [4, 4, 4], 8, exact_budget=True)
    assert result.used_budget == 8
    assert len(result.selected) == 2


def test_cut_penalty_preserves_one_side_only_when_unary_gain_justifies_cut():
    independent = select_chain_with_cut_penalty([3.0, 0.1, 2.9], [0.0, 0.0], [1, 1, 1], 2, True)
    coherent = select_chain_with_cut_penalty([3.0, 0.1, 2.9], [4.0, 0.0], [1, 1, 1], 2, True)
    assert independent.selected == [0, 2]
    assert coherent.selected == [0, 1]


def test_matched_interaction_masks_form_a_distinct_equal_budget_factorial():
    masks = matched_interaction_masks([0.9, 0.2, 0.8, 0.1, 0.7, 0.6, 0.5, 0.4], 2, 4)
    assert set(masks) == {"both", "left_only", "right_only", "neither"}
    assert all(len(mask) == 4 for mask in masks.values())
    assert len({tuple(mask) for mask in masks.values()}) == 4
    common = set(masks["both"]) & set(masks["left_only"]) & set(masks["right_only"]) & set(masks["neither"])
    assert len(common) == 2
    assert {2, 3}.issubset(masks["both"])
    assert 2 in masks["left_only"] and 3 not in masks["left_only"]
    assert 2 not in masks["right_only"] and 3 in masks["right_only"]
    assert not {2, 3} & set(masks["neither"])
