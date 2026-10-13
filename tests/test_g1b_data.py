from collections import Counter

from bikv.g1b_data import generate_g1b_dataset


def test_g1b_is_paired_stratified_and_independent() -> None:
    samples = generate_g1b_dataset(96, 84, 20261015)
    assert len(samples) == 96
    families: dict[str, list] = {}
    for sample in samples:
        sample.validate()
        families.setdefault(sample.metadata["counterfactual_group"], []).append(sample)
    assert len(families) == 48
    assert all(len(rows) == 2 for rows in families.values())
    assert all(len({row.answer for row in rows}) == 2 for rows in families.values())
    assert min(index for row in samples for index in row.supporting_blocks) == 0
    assert max(index for row in samples for index in row.supporting_blocks) == 83
    conditions = Counter(row.metadata["evidence_condition"] for row in samples)
    assert conditions == {"adjacent": 80, "separated": 16}
    for rows in families.values():
        assert rows[0].supporting_blocks == rows[1].supporting_blocks
        assert rows[0].metadata["evidence_condition"] == rows[1].metadata["evidence_condition"]
        assert rows[0].question == rows[1].question
        assert rows[0].metadata["evidence_order"] == rows[1].metadata["evidence_order"]
        roster_markers = ("Laboratory roster", "Personnel assignment", "Shift register")
        left_non_roster = sorted(block for block in rows[0].blocks if not any(mark in block for mark in roster_markers))
        right_non_roster = sorted(block for block in rows[1].blocks if not any(mark in block for mark in roster_markers))
        assert left_non_roster == right_non_roster
        for row in rows:
            target_lab = str(row.metadata["target_lab"])
            audit_blocks = [block for block in row.blocks if block.startswith(AUDIT_PREFIXES)]
            assert audit_blocks
            assert all(target_lab not in block for block in audit_blocks)
            assert row.metadata["audit_target_lab_mentions"] == 0


def test_g1b_seed_is_reproducible_and_changes_documents() -> None:
    first = generate_g1b_dataset(12, 21, 7)
    repeated = generate_g1b_dataset(12, 21, 7)
    changed = generate_g1b_dataset(12, 21, 8)
    assert [row.to_dict() for row in first] == [row.to_dict() for row in repeated]
    assert [row.blocks for row in first] != [row.blocks for row in changed]


AUDIT_PREFIXES = ("Audit note.", "Maintenance log.", "Archive cross-check.")
