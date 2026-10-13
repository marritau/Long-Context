from __future__ import annotations

import math

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from bikv.model import KVModelBackend
from bikv.packing import CacheLayout, PackedSample
from bikv.presses import MatchedSnapKVPress, RichFeatureCapturePress


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [1]


def packed_sample() -> PackedSample:
    torch.manual_seed(19)
    return PackedSample(
        sample=None,
        context_ids=torch.randint(0, 64, (1, 15)),
        query_ids=torch.randint(0, 64, (1, 2)),
        prompt_tail_ids=torch.tensor([[1]]),
        answer_ids=torch.tensor([[1]]),
        answer_text="x",
        answer_aliases=["x"],
        layout=CacheLayout(2, 2, 4, 3, 1, 15, (4, 3, 4)),
    )


def backend(threshold: int, chunk_size: int = 4) -> KVModelBackend:
    torch.manual_seed(23)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    return KVModelBackend(
        model,
        TokenizerStub(),
        4,
        use_chat_template=False,
        prefill_chunk_size=chunk_size,
        prefill_chunk_threshold=threshold,
    )


def assert_nested_close(left, right) -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            assert_nested_close(left_value, right_value)
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=2e-4, abs_tol=2e-4)
    else:
        assert left == right


def test_chunked_rich_capture_matches_one_shot_features() -> None:
    packed = packed_sample()
    one_shot_backend = backend(threshold=100)
    chunked_backend = backend(threshold=4)
    one_shot = one_shot_backend.capture_rich_query_aware_features(
        packed, RichFeatureCapturePress(layout=packed.layout)
    )
    chunked = chunked_backend.capture_rich_query_aware_features(
        packed, RichFeatureCapturePress(layout=packed.layout)
    )

    assert chunked["capture_mode"] == "causal_block_aligned_chunked_exact"
    assert_nested_close(one_shot["layers"], chunked["layers"])
    assert_nested_close(one_shot["boundary_surprise"], chunked["boundary_surprise"])


def test_chunked_query_aware_snapkv_matches_one_shot_cache() -> None:
    packed = packed_sample()
    one_shot_backend = backend(threshold=100)
    chunked_backend = backend(threshold=4)
    one_shot, _, _ = one_shot_backend._prepare_query_aware_cache(
        packed, MatchedSnapKVPress(retained_tokens=10, window_size=4, kernel_size=3)
    )
    chunked, _, lengths = chunked_backend._prepare_query_aware_cache(
        packed, MatchedSnapKVPress(retained_tokens=10, window_size=4, kernel_size=3)
    )

    assert lengths == [10, 10, 10]
    for expected_layer, actual_layer in zip(one_shot.layers, chunked.layers):
        torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=2e-4, rtol=2e-4)
