from bikv.chain_data import generate_chain_dataset
from bikv.graph import (
    arbitrary_pair_candidates,
    graph_action_features,
    pair_primitives,
    relation_graph_mask,
)


def _capture(n: int) -> dict:
    return {
        "legacy_unary": [float(index) for index in range(n)],
        "blocks": [
            {
                "knorm_head_mean__all": float(index),
                "query_attention_head_mean__all": 1.0 if index in {1, n - 2} else 0.1,
            }
            for index in range(n)
        ],
    }


def test_arbitrary_pair_features_do_not_require_adjacency() -> None:
    blocks = ["plain record"] * 8
    blocks[1] = "technician MIRA uses vault V-1"
    blocks[6] = "vault V-1 uses code AB-123"
    capture = _capture(8)
    pair = pair_primitives(capture, blocks, 1, 6)
    assert pair["is_adjacent"] == 0.0
    assert pair["lexical_shared"] >= 1.0
    features = graph_action_features([1, 3, 6], capture, blocks)
    assert not any("distance_fraction" in name or "is_adjacent" in name for name in features)
    assert any(name.startswith("graph_pair_mean__") for name in features)


def test_candidate_pool_contains_non_adjacent_anchors_and_exact_budget() -> None:
    blocks = [f"record {index}" for index in range(14)]
    blocks[1] = "technician MIRA uses vault V-1"
    blocks[12] = "vault V-1 uses code AB-123"
    candidates = arbitrary_pair_candidates(_capture(14), blocks, budget=6, shortlist_size=10)
    assert candidates
    assert all(len(mask) == 6 and len(set(mask)) == 6 for mask in candidates)
    assert any({1, 12}.issubset(mask) for mask in candidates)


def test_relation_graph_recovers_distributed_synthetic_chains_without_gold() -> None:
    samples = generate_chain_dataset(36, 42, 20261018)
    retained = []
    for sample in samples:
        mask = relation_graph_mask(sample.blocks, sample.question, budget=6)
        retained.append(set(sample.supporting_blocks).issubset(mask))
    assert sum(retained) / len(retained) >= 0.8


def test_pair_permutation_preserves_unary_and_changes_pair_features() -> None:
    blocks = ["query A", "A links B", "B links C", "unrelated"]
    capture = _capture(4)
    original = graph_action_features([0, 1, 2], capture, blocks)
    shuffled = graph_action_features(
        [0, 1, 2], capture, blocks, pair_permutation=[3, 1, 0, 2]
    )
    unary = [name for name in original if name.startswith("unary_mean__")]
    pair = [name for name in original if name.startswith("graph_pair_")]
    assert all(original[name] == shuffled[name] for name in unary)
    assert any(original[name] != shuffled[name] for name in pair)
