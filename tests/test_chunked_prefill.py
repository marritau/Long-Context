import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from bikv.model import KVModelBackend
from bikv.packing import CacheLayout, PackedSample


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return [1]


def test_chunked_prefill_matches_one_shot_causal_cache() -> None:
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    packed = PackedSample(
        sample=None,
        context_ids=torch.randint(0, 64, (1, 12)),
        query_ids=torch.empty((1, 0), dtype=torch.long),
        prompt_tail_ids=torch.tensor([[1]]),
        answer_ids=torch.tensor([[1]]),
        answer_text="x",
        answer_aliases=["x"],
        layout=CacheLayout(0, 0, 4, 3, 0, 12, (4, 4, 4)),
    )
    one_shot = KVModelBackend(
        model, TokenizerStub(), 4, use_chat_template=False, prefill_chunk_threshold=100
    )
    chunked = KVModelBackend(
        model,
        TokenizerStub(),
        4,
        use_chat_template=False,
        prefill_chunk_size=4,
        prefill_chunk_threshold=4,
    )

    expected, _ = one_shot._prefill_document(packed)
    actual, lengths = chunked._prefill_document(packed)

    assert lengths == [12, 12]
    for expected_layer, actual_layer in zip(expected.layers, actual.layers):
        torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=1e-5, rtol=1e-5)
