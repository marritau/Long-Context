from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from bikv.features import extract_layer_features, zscore
from bikv.packing import CacheLayout
from bikv.predictor import LinearInteractionModel
from bikv.selector import SelectionResult, select_chain_with_cut_penalty

try:
    from kvpress.presses.base_press import BasePress
    from kvpress.presses.knorm_press import KnormPress
except ImportError as error:  # pragma: no cover - exercised only before dependencies are installed
    raise ImportError(
        "Could not import kvpress==0.5.5. On Python 3.13, install fire==0.7.1 "
        f"after kvpress. Original error: {error}"
    ) from error


def _selected_indices(layout: CacheLayout, selected_blocks: list[int], device: torch.device) -> torch.Tensor:
    indices = list(range(layout.prefix_length))
    for block in sorted(set(selected_blocks)):
        if not 0 <= block < layout.n_blocks:
            raise ValueError(f"Block {block} is outside the layout")
        start = layout.data_start + block * layout.block_size
        indices.extend(range(start, start + layout.block_size))
    indices.extend(range(layout.data_end, layout.original_length))
    return torch.tensor(indices, dtype=torch.long, device=device)


def _gather_cache(
    keys: torch.Tensor, values: torch.Tensor, layout: CacheLayout, selected_blocks: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.shape[0] != 1:
        raise ValueError("The pilot currently supports batch size 1")
    if keys.shape[2] != layout.original_length:
        raise ValueError(f"Layout length {layout.original_length} does not match KV length {keys.shape[2]}")
    positions = _selected_indices(layout, selected_blocks, keys.device)
    gather = positions.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, keys.shape[-1])
    return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()


@dataclass
class FixedBlockPress(BasePress):
    layout: CacheLayout
    selected_blocks: list[int]
    trace: list[dict[str, Any]] = field(default_factory=list)

    def compress_cache_after_prefill(self, cache) -> None:
        """Apply the same fixed gather after a memory-bounded causal prefill."""
        for layer_index, cache_layer in enumerate(cache.layers):
            keys, values = _gather_cache(
                cache_layer.keys, cache_layer.values, self.layout, self.selected_blocks
            )
            cache_layer.keys = keys
            cache_layer.values = values
            self.trace.append(
                {
                    "layer": layer_index,
                    "selected_blocks": sorted(set(self.selected_blocks)),
                    "original_kv_tokens": int(self.layout.original_length),
                    "retained_kv_tokens": int(keys.shape[2]),
                    "mode": "post_chunked_prefill",
                }
            )

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        compressed = _gather_cache(keys, values, self.layout, self.selected_blocks)
        self.trace.append(
            {
                "layer": int(module.layer_idx),
                "selected_blocks": sorted(set(self.selected_blocks)),
                "original_kv_tokens": int(keys.shape[2]),
                "retained_kv_tokens": int(compressed[0].shape[2]),
            }
        )
        return compressed


@dataclass
class ProtectedKnormPress(BasePress):
    """Per-head, per-layer KNorm with a protected prompt and exact document budget."""

    layout: CacheLayout
    retained_document_tokens: int
    uncompressed_layers: int = 2
    trace: list[dict[str, Any]] = field(default_factory=list)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer = int(module.layer_idx)
        if layer < self.uncompressed_layers:
            self.trace.append(
                {
                    "layer": layer,
                    "mode": "uncompressed_early_layer",
                    "original_kv_tokens": int(keys.shape[2]),
                    "retained_kv_tokens": int(keys.shape[2]),
                    "block_retention": [1.0] * self.layout.n_blocks,
                }
            )
            return keys, values
        budget = min(self.retained_document_tokens, self.layout.data_end - self.layout.data_start)
        score_dtype = torch.promote_types(keys.dtype, torch.float32)
        document_scores = -torch.linalg.vector_norm(
            keys[:, :, self.layout.data_start : self.layout.data_end],
            dim=-1,
            dtype=score_dtype,
        )
        document_indices = document_scores.topk(budget, dim=-1).indices + self.layout.data_start
        prefix = torch.arange(self.layout.prefix_length, device=keys.device).view(1, 1, -1)
        prefix = prefix.expand(keys.shape[0], keys.shape[1], -1)
        indices = torch.cat([prefix, document_indices], dim=-1)
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])

        relative = document_indices - self.layout.data_start
        block_retention = []
        for block in range(self.layout.n_blocks):
            start = block * self.layout.block_size
            content_length = int(self.layout.block_content_lengths[block])
            content_end = start + content_length
            retained_fraction = (
                ((relative >= start) & (relative < content_end)).float().sum(dim=-1)
                / max(1, content_length)
            ).mean().item()
            block_retention.append(float(retained_fraction))
        self.trace.append(
            {
                "layer": layer,
                "mode": "protected_token_knorm",
                "original_kv_tokens": int(keys.shape[2]),
                "retained_kv_tokens": int(indices.shape[-1]),
                "retained_document_tokens": int(budget),
                "block_retention": block_retention,
            }
        )
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()


@dataclass
class MatchedSnapKVPress(BasePress):
    """KVPress/SnapKV scoring with an explicit matched total-token budget."""

    retained_tokens: int
    window_size: int = 64
    kernel_size: int = 5
    trace: list[dict[str, Any]] = field(default_factory=list)
    supports_chunked_query_aware: bool = True

    @contextmanager
    def force_final_observation_compression(self, model):
        """Apply SnapKV on a final causal observation chunk even when a past cache exists."""
        self.post_init_from_model(model)

        def hook(module, _inputs, kwargs, output):
            cache = kwargs["past_key_values"]
            cache_layer = cache.layers[module.layer_idx]
            keys, values = cache_layer.keys, cache_layer.values
            keys, values = self.compress(
                module, kwargs["hidden_states"], keys, values, output[1], kwargs
            )
            cache_layer.keys = keys
            cache_layer.values = values
            return output

        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        hooks = [
            layer.self_attn.register_forward_hook(hook, with_kwargs=True)
            for layer in language_model.layers
        ]
        try:
            yield
        finally:
            for registered in hooks:
                registered.remove()

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        key_length = int(keys.shape[2])
        if key_length <= self.window_size:
            raise ValueError("SnapKV prefill must be longer than its observation window")
        budget = min(int(self.retained_tokens), key_length)
        if budget < self.window_size:
            raise ValueError("Matched SnapKV budget must retain its complete observation window")
        window_attention = _window_attention(
            module,
            hidden_states,
            keys,
            self.window_size,
            kwargs["position_embeddings"],
        )
        scores = window_attention.mean(dim=-2)
        scores = F.avg_pool1d(
            scores,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        groups = module.config.num_attention_heads // module.config.num_key_value_heads
        scores = scores.view(
            keys.shape[0],
            module.config.num_key_value_heads,
            groups,
            key_length - self.window_size,
        ).mean(dim=2)
        scores = F.pad(scores, (0, self.window_size), value=float(scores.max().item()) + 1.0)
        indices = scores.topk(budget, dim=-1).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        self.trace.append(
            {
                "layer": int(module.layer_idx),
                "mode": "kvpress_snapkv_matched_budget",
                "original_kv_tokens": key_length,
                "retained_kv_tokens": budget,
                "window_size": self.window_size,
                "kernel_size": self.kernel_size,
            }
        )
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()


@dataclass
class MatchedKnormPress(BasePress):
    """Upstream KNorm score with an explicit matched total-token budget."""

    retained_tokens: int
    trace: list[dict[str, Any]] = field(default_factory=list)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        del hidden_states, attentions, kwargs
        budget = min(int(self.retained_tokens), int(keys.shape[2]))
        score_dtype = torch.promote_types(keys.dtype, torch.float32)
        scores = -torch.linalg.vector_norm(keys, dim=-1, dtype=score_dtype)
        indices = scores.topk(budget, dim=-1).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        self.trace.append(
            {
                "layer": int(module.layer_idx),
                "mode": "kvpress_knorm_matched_budget",
                "original_kv_tokens": int(keys.shape[2]),
                "retained_kv_tokens": budget,
            }
        )
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()


@dataclass
class FeatureCapturePress(BasePress):
    layout: CacheLayout
    records: list[dict[str, Any]] = field(default_factory=list)
    base_press: Any = field(default_factory=lambda: KnormPress(compression_ratio=0.0))
    n_layers: int = 1

    def post_init_from_model(self, model):
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.n_layers = len(language_model.layers)
        self.base_press.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        features = extract_layer_features(
            hidden_states, scores, self.layout, int(module.layer_idx), self.n_layers
        )
        self.records.append(features.to_dict())
        return keys, values


def _distribution_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "max", "q75", "q90")}
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_q75": float(torch.quantile(values, 0.75).item()),
        f"{prefix}_q90": float(torch.quantile(values, 0.90).item()),
    }


def _query_states(module, hidden_states: torch.Tensor, position_embeddings) -> torch.Tensor:
    """Project and rotate Qwen/Llama queries without relying on KVPress internals."""
    if not hasattr(module, "q_proj"):
        raise NotImplementedError(f"Rich capture does not support {module.__class__.__name__}")
    batch_size, query_length, _ = hidden_states.shape
    queries = module.q_proj(hidden_states)
    queries = queries.view(
        batch_size, query_length, module.config.num_attention_heads, module.head_dim
    ).transpose(1, 2)
    cos, sin = position_embeddings
    return (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))


def _window_attention(
    module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    position_embeddings,
) -> torch.Tensor:
    """SnapKV-compatible attention from a trailing query window to earlier keys."""
    key_length = keys.shape[2]
    cos, sin = position_embeddings
    queries = _query_states(
        module,
        hidden_states[:, -window_size:],
        (cos[:, -window_size:], sin[:, -window_size:]),
    )
    key_states = repeat_kv(
        keys, module.config.num_attention_heads // module.config.num_key_value_heads
    )
    scale = float(getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim)))
    logits = torch.matmul(queries.float(), key_states.float().transpose(2, 3)) * scale
    query_positions = torch.arange(
        key_length - window_size, key_length, device=keys.device
    ).view(1, 1, -1, 1)
    key_positions = torch.arange(key_length, device=keys.device).view(1, 1, 1, -1)
    logits = logits.masked_fill(key_positions > query_positions, float("-inf"))
    return torch.softmax(logits, dim=-1)[..., :-window_size]


@dataclass
class RichFeatureCapturePress(BasePress):
    """Capture head- and layer-resolved query-aware diagnostics without pruning."""

    layout: CacheLayout
    records: list[dict[str, Any]] = field(default_factory=list)
    base_press: Any = field(default_factory=lambda: KnormPress(compression_ratio=0.0))

    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def _later_to_earlier_attention(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict[str, Any],
        boundary: int,
    ) -> torch.Tensor:
        left_start = self.layout.data_start + boundary * self.layout.block_size
        left_end = left_start + int(self.layout.block_content_lengths[boundary])
        right_start = self.layout.data_start + (boundary + 1) * self.layout.block_size
        right_end = right_start + int(self.layout.block_content_lengths[boundary + 1])
        cos, sin = kwargs["position_embeddings"]
        queries = _query_states(
            module,
            hidden_states[:, right_start:right_end],
            (cos[:, right_start:right_end], sin[:, right_start:right_end]),
        )
        key_states = repeat_kv(
            keys[:, :, :right_end],
            module.config.num_attention_heads // module.config.num_key_value_heads,
        )
        scale = float(getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim)))
        logits = torch.matmul(queries.float(), key_states.float().transpose(2, 3)) * scale
        query_positions = torch.arange(right_start, right_end, device=keys.device).view(1, 1, -1, 1)
        key_positions = torch.arange(right_end, device=keys.device).view(1, 1, 1, -1)
        logits = logits.masked_fill(key_positions > query_positions, float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        return attention[..., left_start:left_end].sum(dim=-1).mean(dim=-1)[0]

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.layout.suffix_length <= 0:
            raise ValueError("Rich query-aware capture requires a non-empty diagnostic query suffix")
        knorm_scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        query_attention = _window_attention(
            module,
            hidden_states,
            keys,
            self.layout.suffix_length,
            kwargs["position_embeddings"],
        ).float()

        block_rows: list[dict[str, float]] = []
        block_attention_by_head: list[torch.Tensor] = []
        block_key_means: list[torch.Tensor] = []
        block_hidden_means: list[torch.Tensor] = []
        for block in range(self.layout.n_blocks):
            start = self.layout.data_start + block * self.layout.block_size
            end = start + int(self.layout.block_content_lengths[block])
            head_knorm = knorm_scores[0, :, start:end].float().mean(dim=-1)
            head_attention = query_attention[0, :, :, start:end].sum(dim=-1).mean(dim=-1)
            block_attention_by_head.append(head_attention)
            block_key_means.append(keys[0, :, start:end].float().mean(dim=1))
            block_hidden_means.append(hidden_states[0, start:end].float().mean(dim=0))
            row = {}
            row.update(_distribution_stats(head_knorm, "knorm_head"))
            row.update(_distribution_stats(head_attention, "query_attention_head"))
            block_rows.append(row)

        masses = torch.stack(block_attention_by_head, dim=-1)
        probabilities = masses / masses.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_stats = _distribution_stats(entropy, "query_block_entropy_head")

        edge_rows: list[dict[str, float]] = []
        for boundary in range(self.layout.n_blocks - 1):
            cohesion = F.cosine_similarity(
                block_key_means[boundary], block_key_means[boundary + 1], dim=-1
            )
            directed = self._later_to_earlier_attention(
                module, hidden_states, keys, kwargs, boundary
            )
            row = {}
            row.update(_distribution_stats(cohesion, "key_cohesion_head"))
            row.update(_distribution_stats(directed, "later_to_earlier_attention_head"))
            row.update(entropy_stats)
            row["hidden_cosine"] = float(
                F.cosine_similarity(
                    block_hidden_means[boundary].unsqueeze(0),
                    block_hidden_means[boundary + 1].unsqueeze(0),
                    dim=-1,
                ).item()
            )
            row["hidden_jump"] = float(
                torch.linalg.vector_norm(
                    block_hidden_means[boundary] - block_hidden_means[boundary + 1]
                ).item()
                / math.sqrt(block_hidden_means[boundary].numel())
            )
            edge_rows.append(row)

        self.records.append(
            {
                "layer_index": int(module.layer_idx),
                "blocks": block_rows,
                "edges": edge_rows,
            }
        )
        return keys, values


@dataclass
class ChunkedRichFeatureAccumulator:
    """Capture the exact rich statistics while causal prefill is split into aligned chunks."""

    layout: CacheLayout
    records: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "idle"
    chunk_start: int = 0
    block_rows: dict[int, dict[int, dict[str, float]]] = field(default_factory=dict)
    edge_rows: dict[int, dict[int, dict[str, float]]] = field(default_factory=dict)
    entropy_rows: dict[int, dict[str, float]] = field(default_factory=dict)
    hidden_means: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)

    def set_chunk(self, phase: str, start: int) -> None:
        self.phase = phase
        self.chunk_start = int(start)

    def _observe_blocks(self, module, hidden_states, keys, kwargs) -> None:
        layer = int(module.layer_idx)
        q_len = int(hidden_states.shape[1])
        chunk_end = self.chunk_start + q_len
        cos, sin = kwargs["position_embeddings"]
        blocks = self.block_rows.setdefault(layer, {})
        edges = self.edge_rows.setdefault(layer, {})
        hidden_means = self.hidden_means.setdefault(layer, {})
        score_dtype = torch.promote_types(keys.dtype, torch.float32)
        for block in range(self.layout.n_blocks):
            start = self.layout.data_start + block * self.layout.block_size
            content_length = int(self.layout.block_content_lengths[block])
            end = start + content_length
            if start < self.chunk_start or end > chunk_end:
                continue
            local_start = start - self.chunk_start
            local_end = end - self.chunk_start
            head_knorm = -torch.linalg.vector_norm(
                keys[0, :, start:end], dim=-1, dtype=score_dtype
            ).mean(dim=-1)
            row = {}
            row.update(_distribution_stats(head_knorm, "knorm_head"))
            blocks[block] = row
            right_hidden = hidden_states[0, local_start:local_end].float().mean(dim=0)
            hidden_means[block] = right_hidden

            if block == 0:
                continue
            left = block - 1
            left_start = self.layout.data_start + left * self.layout.block_size
            left_end = left_start + int(self.layout.block_content_lengths[left])
            left_key_mean = keys[0, :, left_start:left_end].float().mean(dim=1)
            right_key_mean = keys[0, :, start:end].float().mean(dim=1)
            cohesion = F.cosine_similarity(left_key_mean, right_key_mean, dim=-1)

            queries = _query_states(
                module,
                hidden_states[:, local_start:local_end],
                (cos[:, local_start:local_end], sin[:, local_start:local_end]),
            )
            key_states = repeat_kv(
                keys[:, :, :end], module.config.num_attention_heads // module.config.num_key_value_heads
            )
            scale = float(getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim)))
            logits = torch.matmul(queries.float(), key_states.float().transpose(2, 3)) * scale
            query_positions = torch.arange(start, end, device=keys.device).view(1, 1, -1, 1)
            key_positions = torch.arange(end, device=keys.device).view(1, 1, 1, -1)
            attention = torch.softmax(
                logits.masked_fill(key_positions > query_positions, float("-inf")), dim=-1
            )
            directed = attention[..., left_start:left_end].sum(dim=-1).mean(dim=-1)[0]
            left_hidden = hidden_means[left]
            edge = {}
            edge.update(_distribution_stats(cohesion, "key_cohesion_head"))
            edge.update(_distribution_stats(directed, "later_to_earlier_attention_head"))
            edge["hidden_cosine"] = float(
                F.cosine_similarity(left_hidden.unsqueeze(0), right_hidden.unsqueeze(0), dim=-1).item()
            )
            edge["hidden_jump"] = float(
                torch.linalg.vector_norm(left_hidden - right_hidden).item()
                / math.sqrt(left_hidden.numel())
            )
            edges[left] = edge

    def _observe_query(self, module, hidden_states, keys, kwargs) -> None:
        layer = int(module.layer_idx)
        q_len = int(hidden_states.shape[1])
        cos, sin = kwargs["position_embeddings"]
        queries = _query_states(module, hidden_states, (cos, sin))
        key_states = repeat_kv(
            keys, module.config.num_attention_heads // module.config.num_key_value_heads
        )
        scale = float(getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim)))
        logits = torch.matmul(queries.float(), key_states.float().transpose(2, 3)) * scale
        query_positions = torch.arange(
            self.chunk_start, self.chunk_start + q_len, device=keys.device
        ).view(1, 1, -1, 1)
        key_positions = torch.arange(keys.shape[2], device=keys.device).view(1, 1, 1, -1)
        attention = torch.softmax(
            logits.masked_fill(key_positions > query_positions, float("-inf")), dim=-1
        )
        masses = []
        blocks = self.block_rows.setdefault(layer, {})
        for block in range(self.layout.n_blocks):
            start = self.layout.data_start + block * self.layout.block_size
            end = start + int(self.layout.block_content_lengths[block])
            head_attention = attention[0, :, :, start:end].sum(dim=-1).mean(dim=-1)
            masses.append(head_attention)
            blocks[block].update(_distribution_stats(head_attention, "query_attention_head"))
        mass_tensor = torch.stack(masses, dim=-1)
        probabilities = mass_tensor / mass_tensor.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        self.entropy_rows[layer] = _distribution_stats(entropy, "query_block_entropy_head")

    def forward_hook(self, module, _inputs, kwargs, output):
        if self.phase not in {"blocks", "query"}:
            return output
        cache = kwargs["past_key_values"]
        keys = cache.layers[module.layer_idx].keys
        hidden_states = kwargs["hidden_states"]
        if self.phase == "blocks":
            self._observe_blocks(module, hidden_states, keys, kwargs)
        else:
            self._observe_query(module, hidden_states, keys, kwargs)
        return output

    @contextmanager
    def __call__(self, model):
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        hooks = [
            layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
            for layer in language_model.layers
        ]
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def finalize(self) -> list[dict[str, Any]]:
        records = []
        for layer in sorted(self.block_rows):
            blocks = self.block_rows[layer]
            edges = self.edge_rows.get(layer, {})
            if set(blocks) != set(range(self.layout.n_blocks)):
                raise RuntimeError(f"Layer {layer} did not capture every document block")
            if set(edges) != set(range(self.layout.n_blocks - 1)):
                raise RuntimeError(f"Layer {layer} did not capture every document boundary")
            entropy = self.entropy_rows.get(layer)
            if entropy is None:
                raise RuntimeError(f"Layer {layer} did not capture query attention")
            ordered_edges = []
            for boundary in range(self.layout.n_blocks - 1):
                row = dict(edges[boundary])
                row.update(entropy)
                ordered_edges.append(row)
            records.append(
                {
                    "layer_index": layer,
                    "blocks": [blocks[index] for index in range(self.layout.n_blocks)],
                    "edges": ordered_edges,
                }
            )
        self.records = records
        return records


def aggregate_layer_features(
    layers: list[dict[str, Any]], skip_first_layers: int = 0
) -> tuple[list[float], list[dict[str, float]]]:
    """Collapse layer observations into one sample-level plan and one row per boundary."""
    if not layers:
        raise ValueError("At least one captured layer is required")
    ordered = sorted(layers, key=lambda row: int(row["layer_index"]))
    layers = ordered[skip_first_layers:]
    if not layers:
        raise ValueError("skip_first_layers removed every captured layer")
    n_blocks = len(layers[0]["unary"])
    n_edges = n_blocks - 1
    if any(len(layer["unary"]) != n_blocks or len(layer["edges"]) != n_edges for layer in layers):
        raise ValueError("Captured layers disagree on document shape")
    normalized = [zscore([float(value) for value in layer["unary"]]) for layer in layers]
    unary = [sum(row[index] for row in normalized) / len(normalized) for index in range(n_blocks)]
    feature_names = sorted(layers[0]["edges"][0]) if n_edges else []
    edges = [
        {
            name: sum(float(layer["edges"][boundary][name]) for layer in layers) / len(layers)
            for name in feature_names
        }
        for boundary in range(n_edges)
    ]
    return unary, edges


def matched_interaction_masks(
    unary: list[float], boundary: int, budget_blocks: int
) -> dict[str, list[int]]:
    """Construct a budget-matched 2x2 intervention around one boundary.

    Every mask contains the same common background. Two high-unary replacement
    blocks occupy the treatment slots when the corresponding boundary block is
    absent, so the interaction contrast cannot be explained by a larger cache.
    """
    n_blocks = len(unary)
    if not 0 <= boundary < n_blocks - 1:
        raise ValueError("Boundary is outside the document")
    if not 2 <= budget_blocks <= n_blocks - 2:
        raise ValueError("Matched 2x2 interventions require budget_blocks <= n_blocks - 2")
    left, right = boundary, boundary + 1
    ranked = sorted(
        (index for index in range(n_blocks) if index not in {left, right}),
        key=lambda index: (-unary[index], index),
    )
    replacement_left, replacement_right = ranked[:2]
    common = ranked[2 : 2 + budget_blocks - 2]
    masks = {
        "both": sorted(common + [left, right]),
        "left_only": sorted(common + [left, replacement_right]),
        "right_only": sorted(common + [replacement_left, right]),
        "neither": sorted(common + [replacement_left, replacement_right]),
    }
    if any(len(mask) != budget_blocks for mask in masks.values()):
        raise AssertionError("Interaction masks do not have the requested exact budget")
    if len({tuple(mask) for mask in masks.values()}) != 4:
        raise AssertionError("The four interaction interventions must be distinct")
    return masks


def pair_proposal(unary: list[float], boundary: int, budget_blocks: int) -> list[int]:
    """Build the exact-budget mask whose mandatory anchor is one adjacent pair."""
    n_blocks = len(unary)
    if not 0 <= boundary < n_blocks - 1:
        raise ValueError("Boundary is outside the document")
    if not 2 <= budget_blocks <= n_blocks:
        raise ValueError("Pair proposals require a budget between two and n_blocks")
    anchor = {boundary, boundary + 1}
    remainder = sorted(
        (index for index in range(n_blocks) if index not in anchor),
        key=lambda index: unary[index],
        reverse=True,
    )
    return sorted(anchor | set(remainder[: budget_blocks - 2]))


def select_shared_plan(
    layers: list[dict[str, Any]],
    budget_blocks: int,
    mode: str,
    *,
    predictor: LinearInteractionModel | None = None,
    supporting_blocks: list[int] | None = None,
    continuity_weight: float = 1.0,
    seed: int = 0,
    skip_first_layers: int = 0,
) -> tuple[SelectionResult, dict[str, Any]]:
    """Choose one document mask reused by every transformer layer."""
    unary, edge_rows = aggregate_layer_features(layers, skip_first_layers=skip_first_layers)
    n_blocks = len(unary)
    if not 1 <= budget_blocks <= n_blocks:
        raise ValueError("budget_blocks must be between one and the document block count")
    if mode == "random":
        rng = random.Random(seed)
        unary = [rng.random() for _ in unary]
        continuity = [0.0] * (n_blocks - 1)
    elif mode == "singleton":
        continuity = [0.0] * (n_blocks - 1)
    elif mode == "neighbor":
        proposals = [pair_proposal(unary, boundary, budget_blocks) for boundary in range(n_blocks - 1)]
        selected = max(proposals, key=lambda mask: sum(unary[index] for index in mask))
        result = SelectionResult(selected, sum(unary[index] for index in selected), budget_blocks)
        return result, {"unary": unary, "mode": "best_unary_pair_proposal", "proposals": proposals}
    elif mode == "learned":
        if predictor is None:
            raise ValueError("learned mode requires a predictor")
        proposals = [pair_proposal(unary, boundary, budget_blocks) for boundary in range(n_blocks - 1)]
        predictions = [predictor.predict(row) for row in edge_rows]
        boundary = max(range(n_blocks - 1), key=lambda index: predictions[index])
        selected = proposals[boundary]
        result = SelectionResult(selected, float(predictions[boundary]), budget_blocks)
        return result, {
            "unary": unary,
            "mode": "learned_budget_matched_pair_proposal",
            "selected_boundary": boundary,
            "proposal_predictions": predictions,
            "proposals": proposals,
            "edge_features": edge_rows,
        }
    elif mode in {"oracle", "task_oracle"}:
        required = sorted(set(supporting_blocks or []))
        if not required or len(required) > budget_blocks:
            raise ValueError("task oracle requires gold support that fits the budget")
        remaining = sorted(
            (index for index in range(n_blocks) if index not in required),
            key=lambda index: unary[index],
            reverse=True,
        )
        selected = sorted(required + remaining[: budget_blocks - len(required)])
        result = SelectionResult(selected, sum(unary[index] for index in selected), budget_blocks)
        return result, {"unary": unary, "continuity": [0.0] * (n_blocks - 1), "mode": "task_oracle"}
    else:
        raise ValueError(f"Unknown shared-plan mode: {mode}")
    result = select_chain_with_cut_penalty(
        unary,
        continuity,
        [1] * n_blocks,
        budget_blocks,
        exact_budget=True,
    )
    return result, {"unary": unary, "continuity": continuity, "mode": mode, "edge_features": edge_rows}
