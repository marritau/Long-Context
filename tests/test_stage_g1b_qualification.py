from bikv.g1b_data import generate_g1b_dataset, qualification_plan


def test_qualification_plan_needs_no_capture_and_preserves_exact_budget() -> None:
    sample = generate_g1b_dataset(2, 21, 17)[0]
    plan = qualification_plan(sample, budget=6, seed=19)

    assert plan["masks"]["full"] is None
    oracle = plan["masks"]["task_oracle"]
    assert len(oracle) == 6
    assert set(sample.supporting_blocks).issubset(oracle)
    assert plan == qualification_plan(sample, budget=6, seed=19)
