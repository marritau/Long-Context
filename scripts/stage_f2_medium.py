from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from bikv.config import ProjectConfig
from bikv.presses import (
    FixedBlockPress,
    MatchedKnormPress,
    MatchedSnapKVPress,
    ProtectedKnormPress,
    RichFeatureCapturePress,
    pair_proposal,
)
from bikv.runner import initialize_run, load_backend, make_store
from bikv.schema import load_samples
from offline_stage_e import action_features_v2, legacy_unary_from_capture, summarize_capture_v2


def score_model(features: dict[str, float], model: dict[str, Any]) -> float:
    values = np.asarray([features[name] for name in model["feature_names"]], dtype=np.float64)
    standardized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(np.dot(standardized, np.asarray(model["coefficients"])) + model["intercept"])


def top_mask(values: list[float], budget: int) -> list[int]:
    return sorted(sorted(range(len(values)), key=lambda index: (-values[index], index))[:budget])


def oracle_mask(support: list[int], values: list[float], budget: int) -> list[int]:
    selected = set(support)
    for index in sorted(range(len(values)), key=lambda item: (-values[item], item)):
        if len(selected) >= budget:
            break
        selected.add(index)
    return sorted(selected)


def choose_candidate(
    candidates: list[list[int]], capture: dict[str, Any], model: dict[str, Any]
) -> tuple[list[int], list[float]]:
    values = []
    for mask in candidates:
        rich, decomposable = action_features_v2(mask, capture)
        source = rich if model["kind"] == "ranknet" else decomposable
        values.append(score_model(source, model))
    winner = max(range(len(candidates)), key=lambda index: (values[index], tuple(-v for v in candidates[index])))
    return candidates[winner], values


def within_budget(started: float, config: ProjectConfig) -> bool:
    limit = float(config.get("max_gpu_minutes", 240)) * 60
    reserve = float(config.get("shutdown_reserve_minutes", 12)) * 60
    return time.monotonic() - started < max(0.0, limit - reserve)


def load_or_capture(backend, store, packed, sample, skip: int) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = ".json.gz"
    if store.exists("rich_capture", sample.sample_id, suffix):
        raw = store.read_json("rich_capture", sample.sample_id, suffix)
    else:
        press = RichFeatureCapturePress(layout=packed.layout)
        measured = backend.capture_rich_query_aware_features(packed, press)
        raw = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "split": sample.split,
            "variant": sample.variant,
            "evidence_order": sample.metadata["evidence_order"],
            "supporting_blocks": sample.supporting_blocks,
            "feature_schema": "v12_head_layer_query_pair_hidden_v2",
            "layout": packed.layout.to_dict(),
            **measured,
        }
        store.write_json("rich_capture", sample.sample_id, raw, compress=True)
    return raw, summarize_capture_v2(raw, skip)


def plan_sample(sample, raw, capture, frozen, budget: int, seed: int) -> dict[str, Any]:
    # This exactly matches the layer-wise z-scored KNorm proposal generator used
    # to create Stage V10 action labels. F1 used a different raw-mean ranking.
    legacy_knorm = legacy_unary_from_capture(raw)
    query = [float(block["query_attention_head_mean__all"]) for block in capture["blocks"]]
    candidates = []
    for boundary in range(len(sample.blocks) - 1):
        candidate = pair_proposal(legacy_knorm, boundary, budget)
        if candidate not in candidates:
            candidates.append(candidate)
    sample_seed = seed + int(hashlib.sha1(sample.sample_id.encode()).hexdigest()[:8], 16)
    random_mask = sorted(random.Random(sample_seed).sample(range(len(sample.blocks)), budget))
    model_map = {
        "candidate_query_unary_v2": "query_unary_ranknet_v2",
        "candidate_matched_unary_ridge_v2": "matched_unary_ridge_v2",
        "candidate_pair_compact_v2": "pair_dp_compact_v2",
        "candidate_pair_broad_v2": "pair_dp_broad_v2",
        "candidate_pair_no_query_v2": "pair_dp_no_query_pair_v2",
    }
    chosen = {name: choose_candidate(candidates, capture, frozen[key]) for name, key in model_map.items()}
    if "pair_dp_no_query_pair_v2" in frozen:
        shuffled_capture = dict(capture)
        shuffled_edges = list(capture["edges"])
        random.Random(sample_seed ^ 0x5A17C9E3).shuffle(shuffled_edges)
        shuffled_capture["edges"] = shuffled_edges
        chosen["candidate_pair_shuffled_v2"] = choose_candidate(
            candidates, shuffled_capture, frozen["pair_dp_no_query_pair_v2"]
        )
    masks = {
        "full": None,
        "task_oracle": oracle_mask(sample.supporting_blocks, legacy_knorm, budget),
        "random_block": random_mask,
        "shared_block_knorm": top_mask(legacy_knorm, budget),
        "shared_block_query_attention": top_mask(query, budget),
        **{name: result[0] for name, result in chosen.items()},
        "token_knorm": None,
        "knorm_upstream_query_aware": None,
        "snapkv_upstream_query_aware": None,
    }
    return {
        "legacy_knorm": legacy_knorm,
        "candidate_pool": candidates,
        "masks": masks,
        "policy_scores": {name: result[1] for name, result in chosen.items()},
    }


def evaluate_method(
    backend, packed, sample, method: str, mask, config: ProjectConfig, budget_blocks: int | None = None
):
    max_new_tokens = int(config.get("max_new_tokens", 8))
    budget = int(config.get("budget_blocks")) if budget_blocks is None else int(budget_blocks)
    candidates_text = [str(value) for value in sample.metadata["candidate_codes"]]
    if method in {"knorm_upstream_query_aware", "snapkv_upstream_query_aware"}:
        retained = (
            packed.layout.prefix_length
            + budget * packed.layout.block_size
            + int(packed.query_ids.shape[1])
        )
        if method == "snapkv_upstream_query_aware":
            score_press = MatchedSnapKVPress(
                retained_tokens=retained,
                window_size=int(config.get("snapkv_window_size", 64)),
                kernel_size=int(config.get("snapkv_kernel_size", 5)),
            )
        else:
            score_press = MatchedKnormPress(retained_tokens=retained)
        result = backend.score_candidates_query_aware(packed, candidates_text, score_press)
        if method == "snapkv_upstream_query_aware":
            generation_press = MatchedSnapKVPress(
                retained_tokens=retained,
                window_size=int(config.get("snapkv_window_size", 64)),
                kernel_size=int(config.get("snapkv_kernel_size", 5)),
            )
        else:
            generation_press = MatchedKnormPress(retained_tokens=retained)
        generated = backend.generate_query_aware(packed, generation_press, max_new_tokens=max_new_tokens)
        return result, generated, score_press.trace, retained, "query_in_compressed_prefill"
    if method == "token_knorm":
        press = ProtectedKnormPress(
            layout=packed.layout,
            retained_document_tokens=budget * packed.layout.block_size,
            uncompressed_layers=int(config.get("knorm_uncompressed_layers", 2)),
        )
    elif mask is not None:
        press = FixedBlockPress(layout=packed.layout, selected_blocks=mask)
    else:
        press = None
    result = backend.score_candidates(packed, candidates_text, press=press)
    generated = backend.generate(packed, press=press, max_new_tokens=max_new_tokens)
    trace = press.trace if press is not None else []
    return result, generated, trace, None, "document_compress_then_query"


def write_result(backend, store, packed, sample, method, plan, config):
    mask = plan["masks"][method]
    result, generated, trace, matched_tokens, protocol = evaluate_method(
        backend, packed, sample, method, mask, config
    )
    payload = {
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "counterfactual_group": sample.metadata["counterfactual_group"],
        "split": sample.split,
        "evidence_order": sample.metadata["evidence_order"],
        "method": method,
        "selected_blocks": mask,
        "supporting_blocks": sample.supporting_blocks,
        "support_retained": None if method in {
            "token_knorm", "knorm_upstream_query_aware", "snapkv_upstream_query_aware"
        } else float(
            mask is None or set(sample.supporting_blocks).issubset(mask)
        ),
        "candidate_result": result,
        "generation_result": generated.to_dict(),
        "trace": trace,
        "candidate_pool_size": len(plan["candidate_pool"]) if method.startswith("candidate_") else None,
        "candidate_policy_scores": plan["policy_scores"].get(method),
        "cache_protocol": protocol,
        "query_diagnostic_cache_reused": False,
        "matched_target_kv_tokens": matched_tokens,
        "candidate_generator": "v10_exact_layer_zscored_knorm",
    }
    store.write_json(f"evaluation/{method}", sample.sample_id, payload)
    return result, generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage F2: preregistered medium-context GPU comparison")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    store = make_store(config)
    initialize_run(config, store)
    frozen = json.loads(config.resolve("frozen_models_path").read_text(encoding="utf-8"))
    required = {
        "query_unary_ranknet_v2",
        "pair_dp_compact_v2",
        "pair_dp_broad_v2",
        "pair_dp_no_query_pair_v2",
    }
    if not required.issubset(frozen):
        raise ValueError(f"Frozen model file is missing {sorted(required - set(frozen))}")
    splits = set(config.get("stage_f_splits", ["train", "validation"]))
    samples = [
        sample for sample in load_samples(config.resolve("data_path"))
        if sample.split in splits and sample.variant == "neutral"
    ]
    print(
        f"[stage-f2] Loading {config.get('model_name')} for {len(samples)} fresh stories; "
        f"{config.get('n_blocks')}x{config.get('block_size')} document tokens",
        flush=True,
    )
    backend = load_backend(config)
    started = time.monotonic()
    skip = int(config.get("feature_skip_first_layers", 2))
    budget = int(config.get("budget_blocks"))
    seed = int(config.get("seed"))
    plans = {}
    packed_samples = {}
    qualification_methods = ["full", "task_oracle"]
    comparison_methods = [
        "random_block",
        "shared_block_knorm",
        "shared_block_query_attention",
        "candidate_query_unary_v2",
        "candidate_pair_compact_v2",
        "candidate_pair_broad_v2",
        "candidate_pair_no_query_v2",
        "token_knorm",
        "knorm_upstream_query_aware",
        "snapkv_upstream_query_aware",
    ]
    total = len(samples) * (len(qualification_methods) + len(comparison_methods))
    completed = 0
    skipped = 0

    for sample_index, sample in enumerate(samples, 1):
        if not within_budget(started, config):
            store.write_status(stage="stage_f2", state="PAUSED_TIME_BUDGET", completed=completed, skipped=skipped, total=total)
            store.bundle()
            return
        packed = backend.pack(sample)
        packed_samples[sample.sample_id] = packed
        print(f"[stage-f2] qualification {sample_index}/{len(samples)} {sample.sample_id}: capture/plan", flush=True)
        raw, capture = load_or_capture(backend, store, packed, sample, skip)
        plan = plan_sample(sample, raw, capture, frozen, budget, seed)
        plans[sample.sample_id] = plan
        for method in qualification_methods:
            if store.exists(f"evaluation/{method}", sample.sample_id):
                skipped += 1
                continue
            result, generated = write_result(backend, store, packed, sample, method, plan, config)
            completed += 1
            print(
                f"[stage-f2] qualification {completed + skipped}/{total} {sample.sample_id} {method}: "
                f"candidate={result['is_correct']} greedy={generated.is_correct}",
                flush=True,
            )
        store.write_status(stage="stage_f2", state="QUALIFYING", completed=completed, skipped=skipped, total=total)
        store.bundle()

    full_rows = [store.read_json("evaluation/full", sample.sample_id) for sample in samples]
    oracle_rows = [store.read_json("evaluation/task_oracle", sample.sample_id) for sample in samples]
    full_correct = sum(bool(row["candidate_result"]["is_correct"]) for row in full_rows)
    oracle_accuracy = sum(bool(row["candidate_result"]["is_correct"]) for row in oracle_rows) / len(samples)
    gate = {
        "samples": len(samples),
        "full_candidate_accuracy": full_correct / len(samples),
        "full_correct_samples": full_correct,
        "oracle_candidate_accuracy": oracle_accuracy,
        "min_full_correct_samples": int(config.get("stage_f2_min_full_correct_samples", 24)),
        "min_oracle_accuracy": float(config.get("stage_f2_min_oracle_accuracy", 0.85)),
    }
    gate["passed"] = (
        full_correct >= gate["min_full_correct_samples"]
        and oracle_accuracy >= gate["min_oracle_accuracy"]
    )
    store.write_json("", "qualification_gate", gate)
    print(f"[stage-f2] qualification gate: {json.dumps(gate, sort_keys=True)}", flush=True)
    if not gate["passed"]:
        store.write_status(stage="stage_f2", state="REFINE_MEDIUM_BACKBONE", completed=completed, skipped=skipped, total=total)
        bundle = store.bundle()
        print(f"[stage-f2] comparison blocked; bundle: {bundle}", flush=True)
        return

    for sample_index, sample in enumerate(samples, 1):
        packed = packed_samples[sample.sample_id]
        plan = plans[sample.sample_id]
        for method in comparison_methods:
            if store.exists(f"evaluation/{method}", sample.sample_id):
                skipped += 1
                continue
            if not within_budget(started, config):
                store.write_status(stage="stage_f2", state="PAUSED_TIME_BUDGET", completed=completed, skipped=skipped, total=total)
                store.bundle()
                return
            result, generated = write_result(backend, store, packed, sample, method, plan, config)
            completed += 1
            processed = completed + skipped
            elapsed = time.monotonic() - started
            eta = elapsed / max(1, completed) * max(0, total - processed)
            print(
                f"[stage-f2] {processed}/{total} {sample.sample_id} {method}: "
                f"candidate={result['is_correct']} greedy={generated.is_correct}; ETA~{eta / 60:.1f}m",
                flush=True,
            )
        store.write_status(stage="stage_f2", state="RUNNING", completed=completed, skipped=skipped, total=total)
        store.bundle()
    store.write_status(stage="stage_f2", state="COMPLETE", completed=completed, skipped=skipped, total=total)
    bundle = store.bundle()
    print(f"[stage-f2] COMPLETE: {bundle}", flush=True)


if __name__ == "__main__":
    main()
