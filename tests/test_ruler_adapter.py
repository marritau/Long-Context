from bikv.ruler_adapter import RULER_QUERY_MARKER, convert_ruler_row, ruler_metrics


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text.split()))))


def test_ruler_conversion_keeps_exact_pretokenized_blocks() -> None:
    row = {
        "index": 3,
        "input": f"one two three four five {RULER_QUERY_MARKER} find variables",
        "outputs": ["ABCDE", "FGHIJ"],
        "length": 11,
    }
    sample = convert_ruler_row(row, TokenizerStub(), block_size=3, length_label=4096, source_revision="abc")
    assert sample.metadata["block_token_ids"] == [[0, 1, 2], [3, 4]]
    assert sample.question == "find variables"
    assert sample.answer == "ABCDE FGHIJ"


def test_ruler_metrics_penalize_missing_and_extra_variables() -> None:
    exact = ruler_metrics("ABCDE FGHIJ", ["ABCDE", "FGHIJ"])
    partial = ruler_metrics("ABCDE KLMNO", ["ABCDE", "FGHIJ"])
    assert exact["set_exact_match"] is True
    assert exact["set_f1"] == 1.0
    assert partial["reference_recall"] == 0.5
    assert partial["set_precision"] == 0.5
    assert partial["set_exact_match"] is False
