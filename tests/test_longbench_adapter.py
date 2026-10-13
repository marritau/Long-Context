from bikv.longbench_adapter import qa_metrics, select_longbench_samples
from bikv.packing import FixedBlockPacker


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))


def test_longbench_metrics_match_token_f1_semantics() -> None:
    result = qa_metrics("The Eiffel Tower", ["Eiffel Tower"])
    assert result["answer_exact_match"] is True
    assert result["answer_f1"] == 1.0
    partial = qa_metrics("Eiffel", ["Eiffel Tower"])
    assert 0.0 < partial["answer_f1"] < 1.0


def test_longbench_selection_is_frozen_and_token_bounded(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = []
    for index in range(6):
        rows.append(
            {
                "_id": f"id-{index}",
                "input": "question",
                "context": " ".join(["word"] * (40 + index)),
                "answers": [f"answer {index}"],
                "length": 5000,
            }
        )
    import json
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    first = select_longbench_samples(
        path, Tokenizer(), block_size=16, count=3, seed=7,
        min_document_tokens=40, max_document_tokens=50,
    )
    second = select_longbench_samples(
        path, Tokenizer(), block_size=16, count=3, seed=7,
        min_document_tokens=40, max_document_tokens=50,
    )
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert all(3 <= len(row.blocks) <= 4 for row in first)


def test_longbench_selection_offset_creates_disjoint_holdout(tmp_path) -> None:
    import json

    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "_id": f"id-{index}",
            "input": "question",
            "context": " ".join(["word"] * 40),
            "answers": [f"answer {index}"],
            "length": 5000,
        }
        for index in range(8)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    first = select_longbench_samples(
        path, Tokenizer(), block_size=16, count=4, seed=7, offset=0,
        min_document_tokens=40, max_document_tokens=50,
    )
    holdout = select_longbench_samples(
        path, Tokenizer(), block_size=16, count=4, seed=7, offset=4,
        min_document_tokens=40, max_document_tokens=50,
    )
    assert {row.sample_id for row in first}.isdisjoint(row.sample_id for row in holdout)


def test_longbench_pretokenized_blocks_stay_in_document_only_cache(tmp_path) -> None:
    import json

    path = tmp_path / "rows.jsonl"
    row = {
        "_id": "natural-qa-1",
        "input": "Which city is requested?",
        "context": "alpha beta gamma delta epsilon zeta eta theta",
        "answers": ["Paris"],
        "length": 8,
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    sample = select_longbench_samples(
        path,
        Tokenizer(),
        block_size=4,
        count=1,
        seed=11,
        min_official_length=0,
        max_official_length=100,
        min_document_tokens=1,
        max_document_tokens=100,
    )[0]
    packed = FixedBlockPacker(Tokenizer(), block_size=4, use_chat_template=False).pack(sample)

    assert packed.layout.n_blocks == 2
    assert packed.layout.data_end == packed.layout.original_length
    assert packed.layout.suffix_length == 0
    assert packed.query_ids.shape[1] > 0
    assert packed.context_ids.shape[1] == packed.layout.prefix_length + 8
    assert packed.layout.block_content_lengths == (4, 4)
