from bikv.data import BOUNDARY_CUE, generate_dataset


def test_variants_preserve_latent_task_and_move_only_the_cue():
    samples = generate_dataset(n_stories=3, n_blocks=10, seed=7)
    assert len(samples) == 9
    for story_index in range(3):
        variants = samples[story_index * 3 : story_index * 3 + 3]
        by_name = {sample.variant: sample for sample in variants}
        assert len({sample.answer for sample in variants}) == 1
        assert len({sample.question for sample in variants}) == 1
        assert len({tuple(sample.supporting_blocks) for sample in variants}) == 1
        support_right = by_name["targeted"].supporting_blocks[1]
        assert by_name["targeted"].blocks[support_right].startswith(BOUNDARY_CUE)
        assert not by_name["sham"].blocks[support_right].startswith(BOUNDARY_CUE)
        assert by_name["sham"].blocks[by_name["sham"].manipulated_boundary + 1].startswith(BOUNDARY_CUE)


def test_splits_are_grouped_by_story():
    samples = generate_dataset(n_stories=9, n_blocks=10, seed=11)
    by_story = {}
    for sample in samples:
        by_story.setdefault(sample.story_id, set()).add(sample.split)
    assert all(len(splits) == 1 for splits in by_story.values())
    assert {sample.split for sample in samples} == {"train", "validation", "test"}


def test_counterfactual_pairs_share_entities_but_change_answer_and_never_cross_splits():
    samples = generate_dataset(n_stories=6, n_blocks=10, seed=13)
    neutral = [sample for sample in samples if sample.variant == "neutral"]
    by_family = {}
    for sample in neutral:
        by_family.setdefault(sample.metadata["counterfactual_group"], []).append(sample)
    assert len(by_family) == 3
    for pair in by_family.values():
        assert len(pair) == 2
        assert len({sample.split for sample in pair}) == 1
        assert len({sample.answer for sample in pair}) == 2
        assert pair[0].metadata["candidate_agents"] == pair[1].metadata["candidate_agents"]
        assert pair[0].metadata["candidate_codes"] == pair[1].metadata["candidate_codes"]
        assert pair[0].question == pair[1].question


def test_each_story_has_six_codes_and_only_three_observed_lab_links():
    sample = generate_dataset(n_stories=1, n_blocks=10, seed=21)[0]
    assert sum("verification code assigned" in block for block in sample.blocks) == 6
    assert sum("technician assigned to laboratory" in block for block in sample.blocks) == 3


def test_every_support_ablation_has_at_least_three_symbolic_hypotheses():
    samples = generate_dataset(n_stories=6, n_blocks=10, seed=31)
    for sample in samples:
        candidates = sample.metadata["symbolic_ablation_candidates"]
        assert all(len(set(values)) >= 3 for values in candidates.values())


def test_counterfactual_family_reverses_causal_evidence_order():
    neutral = [
        sample for sample in generate_dataset(n_stories=6, n_blocks=10, seed=37)
        if sample.variant == "neutral"
    ]
    by_family = {}
    for sample in neutral:
        by_family.setdefault(sample.metadata["counterfactual_group"], []).append(sample)
    for pair in by_family.values():
        assert {sample.metadata["evidence_order"] for sample in pair} == {
            "code_then_roster",
            "roster_then_code",
        }
        for sample in pair:
            left, right = sample.supporting_blocks
            assert {sample.metadata["code_block"], sample.metadata["roster_block"]} == {left, right}


def test_gold_codes_do_not_leak_through_audit_references():
    neutral = [
        sample
        for sample in generate_dataset(n_stories=60, n_blocks=14, seed=20261011)
        if sample.variant == "neutral"
    ]
    for sample in neutral:
        code_block = int(sample.metadata["code_block"])
        without_code = "\n".join(
            block for index, block in enumerate(sample.blocks) if index != code_block
        )
        assert sample.answer not in without_code


def test_gold_answers_are_balanced_across_candidate_indices():
    neutral = [
        sample
        for sample in generate_dataset(n_stories=60, n_blocks=14, seed=20261011)
        if sample.variant == "neutral"
    ]
    indices = [sample.metadata["candidate_codes"].index(sample.answer) for sample in neutral]
    assert {index: indices.count(index) for index in range(6)} == {index: 10 for index in range(6)}
