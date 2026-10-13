from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from bikv.config import ProjectConfig
from bikv.g1b_data import qualification_plan
from bikv.graph import arbitrary_pair_candidates, graph_action_features, relation_graph_mask
from bikv.presses import FixedBlockPress, MatchedSnapKVPress, RichFeatureCapturePress
from bikv.runner import initialize_run, load_backend, make_store
from bikv.ruler_adapter import load_ruler_samples, ruler_metrics
from bikv.schema import load_samples

try:
    from offline_stage_e import summarize_capture_v2
    from stage_f2_medium import evaluate_method, score_model, within_budget
except ModuleNotFoundError:
    from scripts.offline_stage_e import summarize_capture_v2
    from scripts.stage_f2_medium import evaluate_method, score_model, within_budget


GRAPH_METHODS = (
    "relation_graph",
    "graph_matched_unary_ridge_v1",
    "graph_arbitrary_pair_ridge_v1",
    "graph_pair_shuffled_v1",
    "snapkv_upstream_query_aware",
)


def resolve(config: ProjectConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.root / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_texts(sample, tokenizer) -> list[str]:
    token_blocks = sample.metadata.get("block_token_ids")
    if token_blocks is None:
        return list(sample.blocks)
    return [tokenizer.decode(ids, skip_special_tokens=False) for ids in token_blocks]


def load_or_capture(backend, store, packed, sample, skip: int, benchmark: str):
    namespace = f"stage_h/captures/{benchmark}"
    if store.exists(namespace, sample.sample_id, ".json.gz"):
        raw = store.read_json(namespace, sample.sample_id, ".json.gz")
    else:
        press = RichFeatureCapturePress(layout=packed.layout)
        measured = backend.capture_rich_query_aware_features(packed, press)
        raw = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "split": getattr(sample, "split", "external_test"),
            "variant": getattr(sample, "variant", "external"),
            "evidence_order": sample.metadata.get("evidence_order", "not_available"),
            "supporting_blocks": sample.supporting_blocks,
            "feature_schema": "stage_h_arbitrary_pair_v1",
            "layout": packed.layout.to_dict(),
            **measured,
        }
        store.write_json(namespace, sample.sample_id, raw, compress=True)
    return raw, summarize_capture_v2(raw, skip)


def choose(
    candidates: list[list[int]], capture, texts, model, pair_permutation: list[int] | None = None
) -> tuple[list[int], list[float]]:
    values = []
    for mask in candidates:
        features = graph_action_features(
            mask, capture, texts, include_distance=False, pair_permutation=pair_permutation
        )
        values.append(score_model(features, model))
    winner = max(range(len(candidates)), key=lambda index: (values[index], tuple(-v for v in candidates[index])))
    return candidates[winner], values


def graph_plan(sample, capture, texts, frozen, budget: int, seed: int) -> dict[str, Any]:
    candidates = arbitrary_pair_candidates(
        capture,
        texts,
        budget,
        shortlist_size=12,
        question=sample.question,
    )
    unary = choose(candidates, capture, texts, frozen["graph_matched_unary_ridge_v1"])
    pair = choose(candidates, capture, texts, frozen["graph_arbitrary_pair_ridge_v1"])
    sample_seed = seed + int(hashlib.sha1(sample.sample_id.encode()).hexdigest()[:8], 16)
    pair_permutation = list(range(len(texts)))
    random.Random(sample_seed ^ 0x48475241).shuffle(pair_permutation)
    shuffled_pair = choose(
        candidates,
        capture,
        texts,
        frozen["graph_arbitrary_pair_ridge_v1"],
        pair_permutation=pair_permutation,
    )
    masks = {
        "relation_graph": relation_graph_mask(texts, sample.question, budget),
        "graph_matched_unary_ridge_v1": unary[0],
        "graph_arbitrary_pair_ridge_v1": pair[0],
        "graph_pair_shuffled_v1": shuffled_pair[0],
        "snapkv_upstream_query_aware": None,
    }
    return {
        "masks": masks,
        "candidate_pool": candidates,
        "scores": {
            "graph_matched_unary_ridge_v1": unary[1],
            "graph_arbitrary_pair_ridge_v1": pair[1],
            "graph_pair_shuffled_v1": shuffled_pair[1],
        },
    }


def synthetic_namespace(method: str) -> str:
    return f"stage_h/synthetic_4k/{method}"


def run_synthetic(backend, store, config, frozen, started: float) -> None:
    path = resolve(config, str(config.get("stage_h_chain_data")))
    audit = json.loads(resolve(config, str(config.get("stage_h_chain_audit"))).read_text(encoding="utf-8"))
    if sha256(path) != str(audit["sha256"]):
        raise ValueError("Stage H chain dataset hash mismatch")
    samples = load_samples(path)
    budget = int(config.get("budget_blocks", 6))
    seed = int(config.get("seed"))
    skip = int(config.get("feature_skip_first_layers", 2))

    # Qualification is completed for the whole frozen set before any graph method.
    for index, sample in enumerate(samples, 1):
        packed = backend.pack(sample)
        plan = qualification_plan(sample, budget, seed)
        for method in ("full", "task_oracle"):
            namespace = synthetic_namespace(method)
            if store.exists(namespace, sample.sample_id):
                continue
            result, generated, trace, matched, protocol = evaluate_method(
                backend, packed, sample, method, plan["masks"][method], config, budget_blocks=budget
            )
            store.write_json(namespace, sample.sample_id, {
                "benchmark": "stage_h_controlled_chains", "sample_id": sample.sample_id,
                "counterfactual_group": sample.metadata["counterfactual_group"],
                "chain_length": sample.metadata["chain_length"],
                "evidence_condition": sample.metadata["evidence_condition"],
                "method": method, "selected_blocks": plan["masks"][method],
                "supporting_blocks": sample.supporting_blocks,
                "support_retained": float(method == "full" or set(sample.supporting_blocks).issubset(plan["masks"][method])),
                "support_fraction": 1.0 if method == "full" else len(set(sample.supporting_blocks) & set(plan["masks"][method])) / len(sample.supporting_blocks),
                "candidate_result": result, "generation_result": generated.to_dict(),
                "trace": trace, "cache_protocol": protocol, "matched_target_kv_tokens": matched,
            })
            print(f"[stage-h:qualify] {index}/{len(samples)} {method}", flush=True)
    full_rows = [store.read_json(synthetic_namespace("full"), row.sample_id) for row in samples]
    oracle_rows = [store.read_json(synthetic_namespace("task_oracle"), row.sample_id) for row in samples]
    gate = {
        "full_accuracy": sum(row["candidate_result"]["is_correct"] for row in full_rows) / len(samples),
        "oracle_accuracy": sum(row["candidate_result"]["is_correct"] for row in oracle_rows) / len(samples),
    }
    gate["passed"] = gate["full_accuracy"] >= 0.65 and gate["oracle_accuracy"] >= 0.80
    store.write_json("stage_h/gates", "synthetic", gate)

    for index, sample in enumerate(samples, 1):
        if not within_budget(started, config):
            raise TimeoutError("GPU time budget reached during Stage H synthetic comparison")
        packed = backend.pack(sample)
        raw, capture = load_or_capture(backend, store, packed, sample, skip, "synthetic")
        texts = block_texts(sample, backend.tokenizer)
        plan = graph_plan(sample, capture, texts, frozen, budget, seed)
        for method in GRAPH_METHODS:
            namespace = synthetic_namespace(method)
            if store.exists(namespace, sample.sample_id):
                continue
            mask = plan["masks"][method]
            result, generated, trace, matched, protocol = evaluate_method(
                backend, packed, sample, method, mask, config, budget_blocks=budget
            )
            selected = set(mask or [])
            support = set(sample.supporting_blocks)
            store.write_json(namespace, sample.sample_id, {
                "benchmark": "stage_h_controlled_chains", "sample_id": sample.sample_id,
                "counterfactual_group": sample.metadata["counterfactual_group"],
                "chain_length": sample.metadata["chain_length"],
                "evidence_condition": sample.metadata["evidence_condition"],
                "method": method, "selected_blocks": mask,
                "supporting_blocks": sample.supporting_blocks,
                "support_retained": None if method == "snapkv_upstream_query_aware" else float(support.issubset(selected)),
                "support_fraction": None if method == "snapkv_upstream_query_aware" else len(support & selected) / len(support),
                "candidate_result": result, "generation_result": generated.to_dict(),
                "candidate_pool_size": len(plan["candidate_pool"]),
                "trace": trace, "cache_protocol": protocol, "matched_target_kv_tokens": matched,
                "qualification_passed": gate["passed"],
            })
            print(f"[stage-h:synthetic] {index}/{len(samples)} {method}", flush=True)
        if index % int(config.get("bundle_every_samples", 2)) == 0:
            store.bundle()


def ruler_namespace(method: str) -> str:
    return f"stage_h/ruler_4k/{method}"


def run_ruler(backend, store, config, frozen, started: float) -> None:
    manifest = json.loads(resolve(config, str(config.get("ruler_manifest_path"))).read_text(encoding="utf-8"))
    path = resolve(config, str(config.get("ruler_data_path")))
    expected = str(manifest["datasets"]["4096"]["sha256"])
    if sha256(path) != expected:
        raise ValueError("Stage H RULER dataset hash mismatch")
    samples = load_ruler_samples(path, backend.tokenizer, int(config.get("block_size", 96)), 4096, str(manifest["revision"]))
    samples = samples[: int(config.get("stage_h_ruler_samples", 20))]
    budget = int(config.get("budget_blocks", 6))
    seed = int(config.get("seed"))
    skip = int(config.get("feature_skip_first_layers", 2))
    pack_kwargs = {
        "system_instruction": "Track variable assignments in the supplied document and answer the query exactly.",
        "answer_instruction": "Return only the requested variable names separated by spaces.",
    }
    methods = ("full",) + GRAPH_METHODS
    for index, sample in enumerate(samples, 1):
        if not within_budget(started, config):
            raise TimeoutError("GPU time budget reached during Stage H RULER comparison")
        packed = backend.pack(sample, **pack_kwargs)
        raw, capture = load_or_capture(backend, store, packed, sample, skip, "ruler")
        texts = block_texts(sample, backend.tokenizer)
        plan = graph_plan(sample, capture, texts, frozen, budget, seed)
        for method in methods:
            namespace = ruler_namespace(method)
            if store.exists(namespace, sample.sample_id):
                continue
            mask = None if method == "full" else plan["masks"][method]
            if method == "snapkv_upstream_query_aware":
                retained = packed.layout.prefix_length + budget * packed.layout.block_size + int(packed.query_ids.shape[1])
                press = MatchedSnapKVPress(retained_tokens=retained, window_size=64, kernel_size=5)
                generated = backend.generate_query_aware(packed, press, max_new_tokens=64)
                trace, protocol = press.trace, "query_in_compressed_prefill"
            else:
                press = FixedBlockPress(packed.layout, mask) if mask is not None else None
                generated = backend.generate(packed, press=press, max_new_tokens=64)
                trace = press.trace if press is not None else []
                protocol = "document_compress_then_query"
            store.write_json(namespace, sample.sample_id, {
                "benchmark": "NVIDIA/RULER", "task": "variable_tracking",
                "sample_id": sample.sample_id, "counterfactual_group": sample.metadata["counterfactual_group"],
                "method": method, "selected_blocks": mask, "budget_blocks": None if method == "full" else budget,
                "generation_result": generated.to_dict(),
                "ruler_metrics": ruler_metrics(generated.generated_answer or "", sample.metadata["references"]),
                "candidate_pool_size": None if method == "full" else len(plan["candidate_pool"]),
                "trace": trace, "cache_protocol": protocol,
            })
            print(f"[stage-h:ruler] {index}/{len(samples)} {method}", flush=True)
        if index % int(config.get("bundle_every_samples", 2)) == 0:
            store.bundle()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage H arbitrary-pair and chain-aware confirmation")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    store = make_store(config)
    initialize_run(config, store)
    frozen = json.loads(resolve(config, str(config.get("stage_h_frozen_models_path"))).read_text(encoding="utf-8"))
    required = {"graph_matched_unary_ridge_v1", "graph_arbitrary_pair_ridge_v1"}
    if not required.issubset(frozen):
        raise ValueError(f"Stage H frozen models are missing {sorted(required - set(frozen))}")
    backend = load_backend(config)
    started = time.monotonic()
    try:
        if bool(config.get("run_stage_h_synthetic", False)):
            run_synthetic(backend, store, config, frozen, started)
        if bool(config.get("run_stage_h_ruler", False)):
            run_ruler(backend, store, config, frozen, started)
    except TimeoutError as error:
        store.write_status(stage="stage_h", state="PAUSED_TIME_BUDGET", reason=str(error))
        print(f"[stage-h] PAUSED: {store.bundle()}", flush=True)
        return
    store.write_status(stage="stage_h", state="COMPLETE")
    print(f"[stage-h] COMPLETE: {store.bundle()}", flush=True)


if __name__ == "__main__":
    main()
