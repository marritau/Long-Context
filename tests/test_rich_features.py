from types import SimpleNamespace

import torch

from bikv.features import FEATURE_NAMES, summarize_layer_edge_features
from bikv.presses import _window_attention


def _layer(index: int, value: float) -> dict:
    edge = {name: value for name in FEATURE_NAMES}
    edge["boundary_position"] = 0.5
    return {"layer_index": index, "unary": [value, value], "edges": [edge]}


def test_rich_summary_preserves_layer_progression() -> None:
    names, rows = summarize_layer_edge_features([_layer(i, float(i)) for i in range(6)])
    assert len(rows) == 1
    assert "hidden_cosine__late_minus_early" in names
    assert rows[0]["hidden_cosine__early"] == 0.5
    assert rows[0]["hidden_cosine__middle"] == 2.5
    assert rows[0]["hidden_cosine__late"] == 4.5
    assert rows[0]["hidden_cosine__late_minus_early"] == 4.0
    assert rows[0]["boundary_position"] == 0.5


def test_rich_summary_rejects_too_few_layers() -> None:
    try:
        summarize_layer_edge_features([_layer(0, 0.0), _layer(1, 1.0)])
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("Expected a ValueError")


def test_window_attention_matches_direct_causal_formula() -> None:
    module = SimpleNamespace(
        q_proj=torch.nn.Linear(4, 4, bias=False),
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=2),
        head_dim=2,
        scaling=2**-0.5,
    )
    module.q_proj.weight.data.copy_(torch.eye(4))
    hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4) / 10
    keys = torch.arange(24, dtype=torch.float32).view(1, 2, 6, 2) / 20
    cos = torch.ones(1, 6, 2)
    sin = torch.zeros(1, 6, 2)

    actual = _window_attention(module, hidden, keys, 2, (cos, sin))

    queries = hidden[:, -2:].view(1, 2, 2, 2).transpose(1, 2)
    logits = torch.matmul(queries, keys.transpose(2, 3)) * module.scaling
    query_positions = torch.tensor([4, 5]).view(1, 1, 2, 1)
    key_positions = torch.arange(6).view(1, 1, 1, 6)
    expected = torch.softmax(logits.masked_fill(key_positions > query_positions, float("-inf")), dim=-1)
    expected = expected[..., :-2]

    assert actual.shape == (1, 2, 2, 4)
    torch.testing.assert_close(actual, expected)
