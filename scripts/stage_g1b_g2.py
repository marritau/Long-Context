from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from bikv.config import ProjectConfig
from bikv.g1b_data import qualification_plan, select_g1b_family_subset
from bikv.presses import FixedBlockPress, MatchedSnapKVPress, RichFeatureCapturePress
from bikv.protocol import comparison_allowed
from bikv.runner import initialize_run, load_backend, make_store
from bikv.ruler_adapter import load_ruler_samples, ruler_metrics
from bikv.schema import load_samples
try:
    from offline_stage_e import summarize_capture_v2
    from stage_f2_medium import evaluate_method, plan_sample, within_budget
except ModuleNotFoundError:
    from scripts.offline_stage_e import summarize_capture_v2
    from scripts.stage_f2_medium import evaluate_method, plan_sample, within_budget


G1B_QUALIFICATION = ("full", "task_oracle")
G1B_FIXED_METHODS = (
    "random_block",
    "shared_block_knorm",
    "shared_block_query_attention",
    "candidate_query_unary_v2",
    "candidate_matched_unary_ridge_v2",
    "candidate_pair_no_query_v2",
    "candidate_pair_shuffled_v2",
    "snapkv_upstream_query_aware",
)
G1B_RATIO_METHODS = (
    "shared_block_knorm",
    "candidate_matched_unary_ridge_v2",
    "candidate_pair_no_query_v2",
    "snapkv_upstream_query_aware",
)
RULER_METHODS = (
    "full",
    "shared_block_knorm",
    "candidate_query_unary_v2",
    "candidate_matched_unary_ridge_v2",
    "candidate_pair_no_query_v2",
    "candidate_pair_shuffled_v2",
    "snapkv_upstream_query_aware",
)
def resolve(config: ProjectConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_capture(backend, store, packed, sample, skip: int, benchmark: str) -> tuple[dict, dict]:
    namespace = f"captures/{benchmark}"
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
            "feature_schema": "v14_g1b_g2_rich_v2",
            "layout": packed.layout.to_dict(),
            **measured,
        }
        store.write_json(namespace, sample.sample_id, raw, compress=True)
    return raw, summarize_capture_v2(raw, skip)


def write_g1b_result(store, backend, packed, sample, method, plan, config, budget: int, arm: str) -> None:
    mask = plan["masks"][method]
    result, generated, trace, matched_tokens, protocol = evaluate_method(
        backend, packed, sample, method, mask, config, budget_blocks=budget
    )
    payload = {
        "benchmark": "g1b_controlled",
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "counterfactual_group": sample.metadata["counterfactual_group"],
        "evidence_condition": sample.metadata["evidence_condition"],
        "method": method,
        "arm": arm,
        "n_blocks": len(sample.blocks),
        "budget_blocks": budget,
        "retention_ratio": budget / len(sample.blocks),
        "selected_blocks": mask,
        "supporting_blocks": sample.supporting_blocks,
        "support_retained": None if method == "snapkv_upstream_query_aware" else float(
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
    namespace = f"g1b/n{len(sample.blocks):03d}/{arm}/b{budget:03d}/{method}"
    store.write_json(namespace, sample.sample_id, payload)


def g1b_namespace(sample, arm: str, budget: int, method: str) -> str:
    return f"g1b/n{len(sample.blocks):03d}/{arm}/b{budget:03d}/{method}"


def run_g1b(backend, store, config, frozen, started: float) -> dict[int, bool]:
    datasets = {int(key): resolve(config, value) for key, value in config.get("g1b_datasets").items()}
    audit = json.loads(resolve(config, str(config.get("g1b_audit_path"))).read_text(encoding="utf-8"))
    for n_blocks, path in datasets.items():
        expected_hash = str(audit["lengths"][str(n_blocks)]["sha256"])
        if sha256(path) != expected_hash:
            raise ValueError(f"G1b dataset hash mismatch for n_blocks={n_blocks}")
    source_samples_by_length = {n: load_samples(path) for n, path in sorted(datasets.items())}
    expected = int(config.get("g1b_stories_per_length", 96))
    if any(len(samples) != expected for samples in source_samples_by_length.values()):
        raise ValueError("G1b dataset size does not match g1b_stories_per_length")
    evaluation_size = int(config.get("g1b_eval_stories_per_length", expected))
    samples_by_length = {
        n: select_g1b_family_subset(samples, evaluation_size)
        for n, samples in source_samples_by_length.items()
    }
    fixed_budget = int(config.get("budget_blocks", 6))
    skip = int(config.get("feature_skip_first_layers", 2))
    seed = int(config.get("seed"))
    gates: dict[int, bool] = {}
    gate_payload = {
        "source_stories_per_length": expected,
        "evaluated_stories_per_length": evaluation_size,
        "selection": "condition-stratified, position-spanning complete counterfactual families",
        "failed_gate_comparison_mode": (
            "exploratory_rescue" if bool(config.get("g1b_run_failed_gate_as_rescue", False)) else "blocked"
        ),
        "lengths": {},
    }

    for n_blocks, samples in samples_by_length.items():
        for index, sample in enumerate(samples, 1):
            packed = backend.pack(sample)
            plan = qualification_plan(sample, fixed_budget, seed)
            for method in G1B_QUALIFICATION:
                namespace = g1b_namespace(sample, "fixed", fixed_budget, method)
                if store.exists(namespace, sample.sample_id):
                    continue
                if not within_budget(started, config):
                    raise TimeoutError("GPU time budget reached during G1b qualification")
                write_g1b_result(store, backend, packed, sample, method, plan, config, fixed_budget, "fixed")
                print(f"[g1b:qualify] n={n_blocks} {index}/{len(samples)} {sample.sample_id} {method}", flush=True)
            store.write_status(stage="g1b_qualification", state="RUNNING", n_blocks=n_blocks, sample=index)
        full = [store.read_json(g1b_namespace(row, "fixed", fixed_budget, "full"), row.sample_id) for row in samples]
        oracle = [store.read_json(g1b_namespace(row, "fixed", fixed_budget, "task_oracle"), row.sample_id) for row in samples]
        full_accuracy = sum(bool(row["candidate_result"]["is_correct"]) for row in full) / len(full)
        oracle_accuracy = sum(bool(row["candidate_result"]["is_correct"]) for row in oracle) / len(oracle)
        passed = (
            full_accuracy >= float(config.get("g1b_min_full_accuracy", 0.8))
            and oracle_accuracy >= float(config.get("g1b_min_oracle_accuracy", 0.85))
        )
        gates[n_blocks] = passed
        gate_payload["lengths"][str(n_blocks)] = {
            "samples": len(samples), "full_candidate_accuracy": full_accuracy,
            "evidence_oracle_candidate_accuracy": oracle_accuracy, "passed": passed,
            "sample_ids": [sample.sample_id for sample in samples],
        }
        store.write_json("gates", "g1b", gate_payload)
        store.bundle()

    if bool(config.get("g1b_qualification_only", False)):
        state = "QUALIFICATION_PASSED" if all(gates.values()) else "QUALIFICATION_BLOCKED"
        store.write_status(stage="g1b_qualification", state=state)
        store.bundle()
        return gates

    rescue_enabled = bool(config.get("g1b_run_failed_gate_as_rescue", False))
    for n_blocks, samples in samples_by_length.items():
        if not comparison_allowed(gates[n_blocks], rescue_enabled):
            print(f"[g1b] n={n_blocks} blocked by qualification gate", flush=True)
            continue
        if not gates[n_blocks]:
            print(
                f"[g1b] n={n_blocks} running EXPLORATORY_RESCUE after failed full-context gate",
                flush=True,
            )
        arms = [("fixed", fixed_budget, G1B_FIXED_METHODS)]
        if bool(config.get("g1b_run_ratio_arm", True)):
            ratio_budget = max(2, round(n_blocks * float(config.get("g1b_ratio", 3 / 7))))
            if ratio_budget != fixed_budget:
                arms.append(("ratio", ratio_budget, G1B_RATIO_METHODS))
        for index, sample in enumerate(samples, 1):
            packed = backend.pack(sample)
            raw, capture = load_or_capture(backend, store, packed, sample, skip, "g1b")
            for arm, budget, methods in arms:
                plan = plan_sample(sample, raw, capture, frozen, budget, seed)
                for method in methods:
                    namespace = g1b_namespace(sample, arm, budget, method)
                    if store.exists(namespace, sample.sample_id):
                        continue
                    if not within_budget(started, config):
                        raise TimeoutError("GPU time budget reached during G1b comparison")
                    write_g1b_result(store, backend, packed, sample, method, plan, config, budget, arm)
                    print(
                        f"[g1b] n={n_blocks} {index}/{len(samples)} arm={arm} B={budget} method={method}",
                        flush=True,
                    )
            store.write_status(stage="g1b", state="RUNNING", n_blocks=n_blocks, sample=index)
            if index % int(config.get("bundle_every_samples", 2)) == 0:
                store.bundle()
    state = "COMPLETE" if all(gates.values()) else (
        "COMPLETE_WITH_EXPLORATORY_RESCUE" if rescue_enabled else "COMPLETE_WITH_BLOCKED_GATES"
    )
    store.write_status(stage="g1b", state=state)
    store.bundle()
    return gates


def ruler_namespace(length: int, method: str) -> str:
    return f"g2_ruler/vt_{length:05d}/{method}"


def generate_ruler(backend, packed, method: str, mask, config, budget: int):
    max_new = int(config.get("ruler_max_new_tokens", 64))
    if method == "snapkv_upstream_query_aware":
        retained = packed.layout.prefix_length + budget * packed.layout.block_size + int(packed.query_ids.shape[1])
        press = MatchedSnapKVPress(
            retained_tokens=retained,
            window_size=int(config.get("snapkv_window_size", 64)),
            kernel_size=int(config.get("snapkv_kernel_size", 5)),
        )
        result = backend.generate_query_aware(packed, press, max_new_tokens=max_new)
        return result, press.trace, retained, "query_in_compressed_prefill"
    press = FixedBlockPress(layout=packed.layout, selected_blocks=mask) if mask is not None else None
    result = backend.generate(packed, press=press, max_new_tokens=max_new)
    return result, press.trace if press is not None else [], None, "document_compress_then_query"


def write_ruler_result(store, backend, packed, sample, method, plan, config, budget: int) -> None:
    mask = plan["masks"][method]
    generated, trace, matched_tokens, protocol = generate_ruler(backend, packed, method, mask, config, budget)
    metrics = ruler_metrics(generated.generated_answer or "", sample.metadata["references"])
    payload = {
        "benchmark": "NVIDIA/RULER",
        "task": "variable_tracking",
        "sample_id": sample.sample_id,
        "method": method,
        "length_label": sample.metadata["length_label"],
        "official_length": sample.metadata["official_length"],
        "n_blocks": len(sample.blocks),
        "budget_blocks": None if method == "full" else budget,
        "selected_blocks": mask,
        "references": sample.metadata["references"],
        "generation_result": generated.to_dict(),
        "ruler_metrics": metrics,
        "trace": trace,
        "cache_protocol": protocol,
        "matched_target_kv_tokens": matched_tokens,
        "source_revision": sample.metadata["source_revision"],
    }
    store.write_json(ruler_namespace(int(sample.metadata["length_label"]), method), sample.sample_id, payload)


def run_ruler(backend, store, config, frozen, started: float) -> dict[int, bool]:
    manifest_path = resolve(config, str(config.get("ruler_manifest_path")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = str(manifest["revision"])
    datasets = {int(key): resolve(config, value) for key, value in config.get("ruler_datasets").items()}
    for length, path in datasets.items():
        expected_hash = str(manifest["datasets"][str(length)]["sha256"])
        if sha256(path) != expected_hash:
            raise ValueError(f"RULER dataset hash mismatch for length={length}")
    samples_by_length = {
        length: load_ruler_samples(path, backend.tokenizer, int(config.get("block_size")), length, revision)
        for length, path in sorted(datasets.items())
    }
    expected = int(config.get("ruler_samples_per_length", 50))
    if any(len(samples) != expected for samples in samples_by_length.values()):
        raise ValueError("RULER dataset size does not match ruler_samples_per_length")
    budget = int(config.get("ruler_budget_blocks", config.get("budget_blocks", 6)))
    skip = int(config.get("feature_skip_first_layers", 2))
    seed = int(config.get("seed"))
    pack_kwargs = {
        "system_instruction": "Track variable assignments in the supplied document and answer the query exactly.",
        "answer_instruction": "Return only the requested variable names separated by spaces.",
    }
    gates: dict[int, bool] = {}
    gate_payload = {
        "failed_gate_comparison_mode": (
            "exploratory_rescue" if bool(config.get("ruler_run_failed_gate_as_rescue", False)) else "blocked"
        ),
        "lengths": {},
    }

    for length, samples in samples_by_length.items():
        for index, sample in enumerate(samples, 1):
            namespace = ruler_namespace(length, "full")
            if store.exists(namespace, sample.sample_id):
                continue
            if not within_budget(started, config):
                raise TimeoutError("GPU time budget reached during RULER qualification")
            packed = backend.pack(sample, **pack_kwargs)
            write_ruler_result(store, backend, packed, sample, "full", {"masks": {"full": None}}, config, budget)
            print(f"[g2:qualify] length={length} {index}/{len(samples)} full", flush=True)
            store.write_status(stage="g2_ruler_qualification", state="RUNNING", length=length, sample=index)
        rows = [store.read_json(ruler_namespace(length, "full"), row.sample_id) for row in samples]
        recall = sum(float(row["ruler_metrics"]["reference_recall"]) for row in rows) / len(rows)
        exact = sum(bool(row["ruler_metrics"]["set_exact_match"]) for row in rows) / len(rows)
        passed = recall >= float(config.get("ruler_min_full_reference_recall", 0.5))
        gates[length] = passed
        gate_payload["lengths"][str(length)] = {
            "samples": len(samples), "full_reference_recall": recall,
            "full_set_exact_match": exact, "passed": passed,
        }
        store.write_json("gates", "g2_ruler", gate_payload)
        store.bundle()

    if bool(config.get("ruler_qualification_only", False)):
        state = "QUALIFICATION_PASSED" if all(gates.values()) else "QUALIFICATION_BLOCKED"
        store.write_status(stage="g2_ruler_qualification", state=state)
        store.bundle()
        return gates

    rescue_enabled = bool(config.get("ruler_run_failed_gate_as_rescue", False))
    for length, samples in samples_by_length.items():
        if not comparison_allowed(gates[length], rescue_enabled):
            print(f"[g2] length={length} blocked by full-context gate", flush=True)
            continue
        if not gates[length]:
            print(
                f"[g2] length={length} running EXPLORATORY_RESCUE after failed full-context gate",
                flush=True,
            )
        for index, sample in enumerate(samples, 1):
            packed = backend.pack(sample, **pack_kwargs)
            raw, capture = load_or_capture(backend, store, packed, sample, skip, "ruler")
            plan = plan_sample(sample, raw, capture, frozen, budget, seed)
            for method in RULER_METHODS[1:]:
                namespace = ruler_namespace(length, method)
                if store.exists(namespace, sample.sample_id):
                    continue
                if not within_budget(started, config):
                    raise TimeoutError("GPU time budget reached during RULER comparison")
                write_ruler_result(store, backend, packed, sample, method, plan, config, budget)
                print(f"[g2] length={length} {index}/{len(samples)} method={method}", flush=True)
            store.write_status(stage="g2_ruler", state="RUNNING", length=length, sample=index)
            if index % int(config.get("bundle_every_samples", 2)) == 0:
                store.bundle()
    state = "COMPLETE" if all(gates.values()) else (
        "COMPLETE_WITH_EXPLORATORY_RESCUE" if rescue_enabled else "COMPLETE_WITH_BLOCKED_GATES"
    )
    store.write_status(stage="g2_ruler", state=state)
    store.bundle()
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined prospective G1b and external RULER G2 evaluation")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    store = make_store(config)
    initialize_run(config, store)
    frozen = json.loads(config.resolve("frozen_models_path").read_text(encoding="utf-8"))
    required = {"query_unary_ranknet_v2", "matched_unary_ridge_v2", "pair_dp_no_query_pair_v2"}
    if not required.issubset(frozen):
        raise ValueError(f"Frozen model file is missing {sorted(required - set(frozen))}")
    backend = load_backend(config)
    started = time.monotonic()
    run_g1b_enabled = bool(config.get("run_g1b", True))
    run_g2_enabled = bool(config.get("run_g2", True))
    if not run_g1b_enabled and not run_g2_enabled:
        raise ValueError("At least one of run_g1b or run_g2 must be enabled")
    try:
        g1b_gates = run_g1b(backend, store, config, frozen, started) if run_g1b_enabled else {}
        ruler_gates = run_ruler(backend, store, config, frozen, started) if run_g2_enabled else {}
    except TimeoutError as error:
        store.write_status(stage="g1b_g2", state="PAUSED_TIME_BUDGET", reason=str(error))
        print(f"[g1b-g2] PAUSED: {store.bundle()}", flush=True)
        return
    store.write_json("gates", "combined", {"g1b": g1b_gates, "g2_ruler": ruler_gates})
    requested_gates = list(g1b_gates.values()) + list(ruler_gates.values())
    rescue_requested = bool(
        config.get("g1b_run_failed_gate_as_rescue", False)
        or config.get("ruler_run_failed_gate_as_rescue", False)
    )
    state = "COMPLETE" if requested_gates and all(requested_gates) else (
        "COMPLETE_WITH_EXPLORATORY_RESCUE" if rescue_requested else "COMPLETE_WITH_BLOCKED_GATES"
    )
    store.write_status(stage="g1b_g2", state=state)
    print(f"[g1b-g2] COMPLETE: {store.bundle()}", flush=True)


if __name__ == "__main__":
    main()
