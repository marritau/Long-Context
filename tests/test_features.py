import torch

from bikv.features import extract_layer_features
from bikv.packing import CacheLayout


def test_feature_extraction_excludes_padding_tokens():
    layout = CacheLayout(
        prefix_length=1,
        data_start=1,
        block_size=3,
        n_blocks=2,
        suffix_length=1,
        original_length=8,
        block_content_lengths=(1, 1),
    )
    hidden = torch.zeros(1, 8, 2)
    hidden[0, 1] = torch.tensor([1.0, 0.0])
    hidden[0, 2:4] = 1000.0
    hidden[0, 4] = torch.tensor([0.0, 1.0])
    hidden[0, 5:7] = -1000.0
    hidden[0, 7] = torch.tensor([1.0, 1.0])
    token_scores = torch.zeros(1, 1, 8)
    token_scores[0, 0, 1] = 3.0
    token_scores[0, 0, 2:4] = 100.0
    token_scores[0, 0, 4] = 5.0
    token_scores[0, 0, 5:7] = -100.0

    features = extract_layer_features(hidden, token_scores, layout, layer_index=0, n_layers=1)

    assert features.unary == [3.0, 5.0]
    assert abs(features.edges[0]["hidden_cosine"]) < 1e-6

