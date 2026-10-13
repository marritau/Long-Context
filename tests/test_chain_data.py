from collections import Counter

from bikv.chain_data import audit_chain_dataset, generate_chain_dataset


def test_chain_dataset_is_balanced_counterfactual_and_distance_stratified() -> None:
    samples = generate_chain_dataset(36, 42, 20261018)
    audit = audit_chain_dataset(samples)
    assert audit["stories"] == 36
    assert audit["families"] == 18
    assert set(audit["strata"].values()) == {3}
    assert {len(row.supporting_blocks) for row in samples} == {2, 3, 4}
    conditions = Counter(row.metadata["evidence_condition"] for row in samples)
    assert conditions == {"adjacent": 18, "dispersed": 18}
    for row in samples:
        gaps = row.metadata["support_gaps"]
        if row.metadata["evidence_condition"] == "adjacent":
            assert set(gaps) == {1}
        else:
            assert min(gaps) >= 2


def test_chain_dataset_seed_is_reproducible() -> None:
    first = generate_chain_dataset(12, 42, 7)
    repeated = generate_chain_dataset(12, 42, 7)
    changed = generate_chain_dataset(12, 42, 8)
    assert [row.to_dict() for row in first] == [row.to_dict() for row in repeated]
    assert [row.blocks for row in first] != [row.blocks for row in changed]
