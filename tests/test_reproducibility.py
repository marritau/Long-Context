from pearl.experiments.run_experiment import build_smoke_example
from pearl.packing.pack_builder import pearl_pack


def test_pearl_pack_is_deterministic_for_smoke_example():
    example = build_smoke_example()
    config = {
        "chunking": {"chunk_size": 18, "chunk_overlap": 4},
        "retrieval": {"top_m": 8, "top_k": 4, "dense_dim": 64},
        "selection": {"pre_compression_budget": 100},
        "compression": {"final_budget": 60, "min_chunk_budget": 5},
    }

    prompt_a, packed_a = pearl_pack(example, config=config)
    prompt_b, packed_b = pearl_pack(example, config=config)

    assert prompt_a == prompt_b
    assert [chunk.chunk.chunk_id for chunk in packed_a.chunks] == [
        chunk.chunk.chunk_id for chunk in packed_b.chunks
    ]
    assert packed_a.total_tokens == packed_b.total_tokens
