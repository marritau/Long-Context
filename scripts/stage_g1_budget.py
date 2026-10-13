from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

from bikv.config import ProjectConfig
from bikv.runner import initialize_run, load_backend, make_store
from bikv.schema import load_samples
from offline_stage_e import summarize_capture_v2
from stage_f2_medium import evaluate_method, plan_sample, within_budget


METHODS = [
    "random_block",
    "shared_block_knorm",
    "shared_block_query_attention",
    "candidate_query_unary_v2",
    "candidate_pair_no_query_v2",
    "task_oracle",
    "token_knorm",
    "snapkv_upstream_query_aware",
]
SOURCE_METHODS = ["full", *METHODS]


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def validate_source(source: Path, config: ProjectConfig, sample_ids: list[str]) -> None:
    required_files = [source / "status.json", source / "manifest.json", source / "qualification_gate.json"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage F2 source is incomplete; missing {missing}")
    status = read_json(source / "status.json")
    manifest = read_json(source / "manifest.json")
    gate = read_json(source / "qualification_gate.json")
    source_config = manifest.get("config", {})
    checks = {
        "state": status.get("state") == "COMPLETE",
        "qualification": gate.get("passed") is True,
        "model_name": source_config.get("model_name") == config.get("model_name"),
        "seed": int(source_config.get("seed", -1)) == int(config.get("source_seed")),
        "n_blocks": int(source_config.get("n_blocks", -1)) == int(config.get("n_blocks")),
        "block_size": int(source_config.get("block_size", -1)) == int(config.get("block_size")),
        "source_budget": int(source_config.get("budget_blocks", -1)) == int(config.get("source_budget_blocks")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Stage F2 source failed compatibility checks: {failed}")
    for sample_id in sample_ids:
        if not (source / "rich_capture" / f"{sample_id}.json.gz").exists():
            raise FileNotFoundError(f"Missing F2 capture for {sample_id}")
        for method in SOURCE_METHODS:
            if not (source / "evaluation" / method / f"{sample_id}.json").exists():
                raise FileNotFoundError(f"Missing F2 result: {method}/{sample_id}")


def write_evaluation(store, backend, packed, sample, method, plan, config, budget: int) -> None:
    mask = plan["masks"][method]
    result, generated, trace, matched_tokens, protocol = evaluate_method(
        backend, packed, sample, method, mask, config, budget_blocks=budget
    )
    payload = {
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "counterfactual_group": sample.metadata["counterfactual_group"],
        "split": sample.split,
        "evidence_order": sample.metadata["evidence_order"],
        "method": method,
        "budget_blocks": budget,
        "retention_ratio": budget / len(sample.blocks),
        "selected_blocks": mask,
        "supporting_blocks": sample.supporting_blocks,
        "support_retained": None if method in {"token_knorm", "snapkv_upstream_query_aware"} else float(
            set(sample.supporting_blocks).issubset(mask)
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
        "source": "fresh_g1a_evaluation",
    }
    store.write_json(f"evaluation/budget_{budget:02d}/{method}", sample.sample_id, payload)


def import_source_budget(store, source: Path, sample_ids: list[str], budget: int, n_blocks: int) -> int:
    imported = 0
    for sample_id in sample_ids:
        for method in SOURCE_METHODS:
            namespace = f"evaluation/budget_{budget:02d}/{method}"
            if store.exists(namespace, sample_id):
                continue
            payload = read_json(source / "evaluation" / method / f"{sample_id}.json")
            payload.update(
                {
                    "budget_blocks": budget,
                    "retention_ratio": budget / n_blocks,
                    "source": "reused_verified_stage_f2",
                }
            )
            store.write_json(namespace, sample_id, payload)
            imported += 1
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage G1a: frozen-selector budget stress test")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    store = make_store(config)
    initialize_run(config, store)

    budgets = [int(value) for value in config.get("stage_g1_budgets", [6, 10, 18])]
    source_budget = int(config.get("source_budget_blocks", 18))
    if budgets != sorted(set(budgets)) or any(value < 2 or value > int(config.get("n_blocks")) for value in budgets):
        raise ValueError("stage_g1_budgets must be unique, sorted, and between 2 and n_blocks")
    if source_budget not in budgets:
        raise ValueError("source_budget_blocks must be included in stage_g1_budgets")

    frozen = json.loads(config.resolve("frozen_models_path").read_text(encoding="utf-8"))
    required_models = {"query_unary_ranknet_v2", "pair_dp_no_query_pair_v2"}
    if not required_models.issubset(frozen):
        raise ValueError(f"Frozen model file is missing {sorted(required_models - set(frozen))}")
    splits = set(config.get("stage_g1_splits", ["train", "validation"]))
    samples = [
        sample for sample in load_samples(config.resolve("data_path"))
        if sample.split in splits and sample.variant == "neutral"
    ]
    source = Path(str(config.get("source_run_root"))).resolve()
    validate_source(source, config, [sample.sample_id for sample in samples])
    if args.validate_only:
        print(
            f"[stage-g1a] validation passed: {len(samples)} samples, budgets={budgets}, source={source}",
            flush=True,
        )
        return

    imported = import_source_budget(
        store, source, [sample.sample_id for sample in samples], source_budget, int(config.get("n_blocks"))
    )
    store.write_json("", "source_validation", {
        "passed": True,
        "source_run_root": str(source),
        "source_budget_blocks": source_budget,
        "source_rows_imported": imported,
        "samples": len(samples),
    })
    fresh_budgets = [budget for budget in budgets if budget != source_budget]
    total = len(samples) * len(fresh_budgets) * len(METHODS)
    completed = 0
    skipped = 0
    for sample in samples:
        for budget in fresh_budgets:
            for method in METHODS:
                if store.exists(f"evaluation/budget_{budget:02d}/{method}", sample.sample_id):
                    skipped += 1
    print(
        f"[stage-g1a] verified {len(samples)} F2 samples; reused B={source_budget}; "
        f"fresh evaluations={total}, already checkpointed={skipped}",
        flush=True,
    )
    if skipped == total:
        store.write_status(stage="stage_g1a", state="COMPLETE", completed=0, skipped=skipped, total=total)
        print(f"[stage-g1a] COMPLETE: {store.bundle()}", flush=True)
        return

    backend = load_backend(config)
    started = time.monotonic()
    skip_layers = int(config.get("feature_skip_first_layers", 2))
    seed = int(config.get("seed"))
    processed_samples = 0
    for sample_index, sample in enumerate(samples, 1):
        packed = backend.pack(sample)
        raw = read_json(source / "rich_capture" / f"{sample.sample_id}.json.gz")
        capture = summarize_capture_v2(raw, skip_layers)
        for budget in fresh_budgets:
            plan = plan_sample(sample, raw, capture, frozen, budget, seed)
            for method in METHODS:
                namespace = f"evaluation/budget_{budget:02d}/{method}"
                if store.exists(namespace, sample.sample_id):
                    continue
                if not within_budget(started, config):
                    store.write_status(
                        stage="stage_g1a", state="PAUSED_TIME_BUDGET", completed=completed,
                        skipped=skipped, total=total,
                    )
                    print(f"[stage-g1a] PAUSED: {store.bundle()}", flush=True)
                    return
                write_evaluation(store, backend, packed, sample, method, plan, config, budget)
                completed += 1
                done = completed + skipped
                elapsed = time.monotonic() - started
                eta = elapsed / max(1, completed) * max(0, total - done)
                print(
                    f"[stage-g1a] {done}/{total} sample={sample.sample_id} B={budget} method={method}; "
                    f"ETA~{eta / 60:.1f}m",
                    flush=True,
                )
        processed_samples += 1
        store.write_status(stage="stage_g1a", state="RUNNING", completed=completed, skipped=skipped, total=total)
        if processed_samples % int(config.get("bundle_every_samples", 1)) == 0:
            store.bundle()
    store.write_status(stage="stage_g1a", state="COMPLETE", completed=completed, skipped=skipped, total=total)
    print(f"[stage-g1a] COMPLETE: {store.bundle()}", flush=True)


if __name__ == "__main__":
    main()
