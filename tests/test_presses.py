import importlib
import sys
import types

import torch


def _load_presses_with_fake_kvpress(monkeypatch):
    kvpress = types.ModuleType("kvpress")
    presses_package = types.ModuleType("kvpress.presses")
    base_module = types.ModuleType("kvpress.presses.base_press")
    knorm_module = types.ModuleType("kvpress.presses.knorm_press")

    class BasePress:
        pass

    class KnormPress:
        def __init__(self, compression_ratio=0.0):
            self.compression_ratio = compression_ratio

        def post_init_from_model(self, model):
            return None

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            return -torch.linalg.vector_norm(keys.float(), dim=-1)

    base_module.BasePress = BasePress
    knorm_module.KnormPress = KnormPress
    monkeypatch.setitem(sys.modules, "kvpress", kvpress)
    monkeypatch.setitem(sys.modules, "kvpress.presses", presses_package)
    monkeypatch.setitem(sys.modules, "kvpress.presses.base_press", base_module)
    monkeypatch.setitem(sys.modules, "kvpress.presses.knorm_press", knorm_module)
    sys.modules.pop("bikv.presses", None)
    return importlib.import_module("bikv.presses")


def test_shared_plan_press_physically_keeps_exact_document_budget(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.features import FEATURE_NAMES
    from bikv.packing import CacheLayout

    layout = CacheLayout(
        prefix_length=2,
        data_start=2,
        block_size=4,
        n_blocks=4,
        suffix_length=2,
        original_length=20,
    )
    layers = [{
        "layer_index": 0,
        "unary": [4.0, 3.0, 2.0, 1.0],
        "edges": [{name: 0.0 for name in FEATURE_NAMES} for _ in range(3)],
    }]
    plan, _ = presses.select_shared_plan(layers, budget_blocks=2, mode="singleton")
    press = presses.FixedBlockPress(layout=layout, selected_blocks=plan.selected)
    module = types.SimpleNamespace(layer_idx=0)
    hidden = torch.randn(1, 20, 8)
    keys = torch.randn(1, 2, 20, 4)
    values = torch.randn(1, 2, 20, 4)
    output_keys, output_values = press.compress(module, hidden, keys, values, None, {})

    assert output_keys.shape == output_values.shape == (1, 2, 12, 4)
    assert press.trace[0]["selected_blocks"] == plan.selected
    assert press.trace[0]["retained_kv_tokens"] == 2 + 2 * 4 + 2


def test_fixed_press_preserves_requested_blocks_and_prompt(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.packing import CacheLayout

    layout = CacheLayout(1, 1, 3, 3, 1, 11)
    press = presses.FixedBlockPress(layout=layout, selected_blocks=[0, 2])
    module = types.SimpleNamespace(layer_idx=0)
    hidden = torch.randn(1, 11, 6)
    keys = torch.arange(11, dtype=torch.float32).view(1, 1, 11, 1)
    values = keys.clone()
    output, _ = press.compress(module, hidden, keys, values, None, {})
    assert output.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0]


def test_fixed_press_post_chunked_cache_matches_direct_gather(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.packing import CacheLayout

    layout = CacheLayout(1, 1, 3, 3, 0, 10)
    press = presses.FixedBlockPress(layout=layout, selected_blocks=[0, 2])
    keys = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
    direct, _ = press.compress(
        types.SimpleNamespace(layer_idx=0),
        torch.empty(1, 10, 1),
        keys,
        keys.clone(),
        None,
        {},
    )
    cache = types.SimpleNamespace(
        layers=[types.SimpleNamespace(keys=keys.clone(), values=keys.clone())]
    )
    press.trace.clear()
    press.compress_cache_after_prefill(cache)

    torch.testing.assert_close(cache.layers[0].keys, direct)
    assert press.trace[0]["mode"] == "post_chunked_prefill"


def test_protected_knorm_keeps_prompt_and_exact_per_head_token_budget(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.packing import CacheLayout

    layout = CacheLayout(2, 2, 3, 3, 0, 11, (2, 2, 2))
    press = presses.ProtectedKnormPress(
        layout=layout, retained_document_tokens=4, uncompressed_layers=1
    )
    hidden = torch.randn(1, 11, 6)
    keys = torch.arange(22, dtype=torch.float32).view(1, 2, 11, 1)
    values = keys.clone()
    early, _ = press.compress(
        types.SimpleNamespace(layer_idx=0), hidden, keys, values, None, {}
    )
    compressed, _ = press.compress(
        types.SimpleNamespace(layer_idx=1), hidden, keys, values, None, {}
    )
    assert early.shape == (1, 2, 11, 1)
    assert compressed.shape == (1, 2, 6, 1)
    assert compressed[:, :, :2].tolist() == keys[:, :, :2].tolist()
    assert press.trace[-1]["retained_document_tokens"] == 4
    assert len(press.trace[-1]["block_retention"]) == 3


def test_matched_knorm_keeps_exact_total_budget(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    press = presses.MatchedKnormPress(retained_tokens=5)
    module = types.SimpleNamespace(layer_idx=3)
    hidden = torch.randn(1, 9, 6)
    keys = torch.arange(18, dtype=torch.float32).view(1, 2, 9, 1)
    values = keys.clone()
    compressed_keys, compressed_values = press.compress(
        module, hidden, keys, values, None, {}
    )
    assert compressed_keys.shape == compressed_values.shape == (1, 2, 5, 1)
    assert press.trace == [{
        "layer": 3,
        "mode": "kvpress_knorm_matched_budget",
        "original_kv_tokens": 9,
        "retained_kv_tokens": 5,
    }]


def test_shared_task_oracle_always_keeps_gold_evidence(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.features import FEATURE_NAMES

    layers = [
        {
            "layer_index": 0,
            "unary": [10.0, 9.0, 0.0, -1.0],
            "edges": [
                {name: 0.0 for name in FEATURE_NAMES}
                for _ in range(3)
            ],
        }
    ]
    result, details = presses.select_shared_plan(
        layers, budget_blocks=3, mode="task_oracle", supporting_blocks=[2, 3]
    )
    assert {2, 3}.issubset(result.selected)
    assert len(result.selected) == 3
    assert details["mode"] == "task_oracle"


def test_learned_plan_selects_the_highest_scored_exact_budget_proposal(monkeypatch):
    presses = _load_presses_with_fake_kvpress(monkeypatch)
    from bikv.features import FEATURE_NAMES

    layers = [
        {
            "layer_index": 0,
            "unary": [9.0, 8.0, 7.0, 6.0, 5.0],
            "edges": [
                {name: float(boundary) if name == "boundary_position" else 0.0 for name in FEATURE_NAMES}
                for boundary in range(4)
            ],
        }
    ]

    class Predictor:
        def predict(self, row):
            return row["boundary_position"]

    result, details = presses.select_shared_plan(
        layers, budget_blocks=3, mode="learned", predictor=Predictor()
    )
    assert details["selected_boundary"] == 3
    assert {3, 4}.issubset(result.selected)
    assert len(result.selected) == 3
    assert all(len(mask) == 3 for mask in details["proposals"])
