from pearl.data.schema import Chunk, PackedChunk
from pearl.packing.ordering import position_aware_order


def _packed(idx: int) -> PackedChunk:
    chunk = Chunk(
        chunk_id=f"c{idx}",
        doc_id="d",
        text=f"chunk {idx}",
        token_len=2,
        section_title=None,
        chunk_index=idx,
        char_start=idx * 10,
        char_end=idx * 10 + 5,
    )
    return PackedChunk(chunk=chunk, text=chunk.text, token_len=2, original_token_len=2)


def test_pearl_order_places_top_importance_at_edges():
    chunks = [_packed(i) for i in range(4)]
    ordered = position_aware_order(chunks, importances=[0.1, 0.9, 0.2, 0.8], mode="pearl")

    edge_ids = {ordered[0].chunk.chunk_id, ordered[-1].chunk.chunk_id}
    assert edge_ids == {"c1", "c3"}
    assert ordered[0].position == 0.0
    assert ordered[-1].position == 1.0
