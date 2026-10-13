from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from bikv.config import ProjectConfig
from bikv.longbench_adapter import qa_metrics, select_longbench_samples, selection_manifest
from bikv.presses import FixedBlockPress, MatchedSnapKVPress, RichFeatureCapturePress
from bikv.runner import initialize_run, load_backend, make_store

try:
    from offline_stage_e import summarize_capture_v2
    from stage_f2_medium import within_budget
    from stage_h_graph import graph_plan
except ModuleNotFoundError:
    from scripts.offline_stage_e import summarize_capture_v2
    from scripts.stage_f2_medium import within_budget
    from scripts.stage_h_graph import graph_plan


METHODS = (
    "full",
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


def decoded_blocks(sample, tokenizer) -> list[str]:
    return [
        tokenizer.decode(values, skip_special_tokens=False)
        for values in sample.metadata["block_token_ids"]
    ]


def load_or_capture(backend, store, packed, sample, skip: int):
    namespace = "stage_i/captures/longbench_2wiki"
    if store.exists(namespace, sample.sample_id, ".json.gz"):
        raw = store.read_json(namespace, sample.sample_id, ".json.gz")
    else:
        press = RichFeatureCapturePress(layout=packed.layout)
        measured = backend.capture_rich_query_aware_features(packed, press)
        raw = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "feature_schema": "stage_i_longbench_arbitrary_pair_v1",
            "layout": packed.layout.to_dict(),
            **measured,
        }
        store.write_json(namespace, sample.sample_id, raw, compress=True)
    return summarize_capture_v2(raw, skip)


def generate_method(
    backend, packed, method: str, mask, budget: int, max_new_tokens: int,
    snapkv_window_size: int, snapkv_kernel_size: int,
):
    if method == "snapkv_upstream_query_aware":
        retained = (
            packed.layout.prefix_length
            + budget * packed.layout.block_size
            + int(packed.query_ids.shape[1])
        )
        press = MatchedSnapKVPress(
            retained_tokens=retained,
            window_size=snapkv_window_size,
            kernel_size=snapkv_kernel_size,
        )
        generated = backend.generate_query_aware(packed, press, max_new_tokens=max_new_tokens)
        return generated, press.trace, "query_in_compressed_prefill", retained
    press = FixedBlockPress(packed.layout, mask) if mask is not None else None
    generated = backend.generate(packed, press=press, max_new_tokens=max_new_tokens)
    return generated, press.trace if press else [], "document_compress_then_query", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I natural-data transfer on LongBench-E 2Wiki")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    store = make_store(config)
    initialize_run(config, store)

    source = config.resolve("data_path")
    source_manifest_path = resolve(config, str(config.get("longbench_source_manifest")))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256(source) != source_manifest["sha256"]:
        raise ValueError("LongBench source hash mismatch")
    frozen = json.loads(
        resolve(config, str(config.get("stage_h_frozen_models_path"))).read_text(encoding="utf-8")
    )
    backend = load_backend(config)
    samples = select_longbench_samples(
        source,
        backend.tokenizer,
        block_size=int(config.get("block_size")),
        count=int(config.get("longbench_samples", 24)),
        seed=int(config.get("seed")),
        offset=int(config.get("longbench_selection_offset", 0)),
        min_official_length=int(config.get("longbench_min_official_length", 0)),
        max_official_length=int(config.get("longbench_max_official_length", 1000000)),
        min_document_tokens=int(config.get("longbench_min_document_tokens", 3072)),
        max_document_tokens=int(config.get("longbench_max_document_tokens", 6144)),
    )
    selected = {
        **selection_manifest(samples),
        "source_sha256": source_manifest["sha256"],
        "source_revision": source_manifest["revision"],
        "selection_seed": int(config.get("seed")),
        "selection_offset": int(config.get("longbench_selection_offset", 0)),
    }
    prior_manifest_value = config.get("longbench_prior_selection_manifest")
    if prior_manifest_value:
        prior_manifest = json.loads(resolve(config, str(prior_manifest_value)).read_text(encoding="utf-8"))
        overlap = set(selected["sample_ids"]) & set(prior_manifest["sample_ids"])
        if overlap:
            raise ValueError(f"LongBench holdout overlaps prior cohort: {sorted(overlap)}")
    expected_manifest_value = config.get("longbench_expected_selection_manifest")
    if expected_manifest_value:
        expected_manifest = json.loads(
            resolve(config, str(expected_manifest_value)).read_text(encoding="utf-8")
        )
        for key in ("samples", "sample_ids", "document_tokens", "block_counts"):
            if selected[key] != expected_manifest[key]:
                raise ValueError(f"Frozen LongBench holdout changed for {key}")
    if store.exists("stage_i", "selection_manifest"):
        if store.read_json("stage_i", "selection_manifest") != selected:
            raise ValueError("Frozen LongBench selection changed during resume")
    else:
        store.write_json("stage_i", "selection_manifest", selected)
    selected_cohort = {
        "source_sha256": source_manifest["sha256"],
        "source_revision": source_manifest["revision"],
        "selection_seed": int(config.get("seed")),
        "selection_offset": int(config.get("longbench_selection_offset", 0)),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "source_id": sample.metadata["source_id"],
                "context": sample.metadata["source_context"],
                "question": sample.question,
                "references": sample.metadata["references"],
                "document_tokens": sample.metadata["document_tokens"],
                "official_length_words": sample.metadata["official_length_words"],
            }
            for sample in samples
        ],
    }
    if store.exists("stage_i", "selected_cohort"):
        if store.read_json("stage_i", "selected_cohort") != selected_cohort:
            raise ValueError("Frozen LongBench cohort changed during resume")
    else:
        store.write_json("stage_i", "selected_cohort", selected_cohort)

    budget = int(config.get("budget_blocks"))
    max_new_tokens = int(config.get("max_new_tokens", 32))
    snapkv_window_size = int(config.get("snapkv_window_size", 64))
    snapkv_kernel_size = int(config.get("snapkv_kernel_size", 5))
    skip = int(config.get("feature_skip_first_layers", 2))
    seed = int(config.get("seed"))
    started = time.monotonic()
    total = len(samples) * len(METHODS)
    completed = 0
    skipped = 0
    pack_kwargs = {
        "system_instruction": "Answer the question using only the supplied passages.",
        "answer_instruction": "Only give the answer. Do not provide any explanation.",
    }

    for sample_index, sample in enumerate(samples, 1):
        if not within_budget(started, config):
            store.write_status(
                stage="stage_i_longbench", state="PAUSED_TIME_BUDGET",
                completed=completed, skipped=skipped, total=total,
            )
            print(f"[stage-i] PAUSED: {store.bundle()}", flush=True)
            return
        packed = backend.pack(sample, **pack_kwargs)
        capture = load_or_capture(backend, store, packed, sample, skip)
        texts = decoded_blocks(sample, backend.tokenizer)
        plan = graph_plan(sample, capture, texts, frozen, budget, seed)
        for method in METHODS:
            namespace = f"stage_i/longbench_2wiki/{method}"
            if store.exists(namespace, sample.sample_id):
                skipped += 1
                continue
            mask = None if method == "full" else plan["masks"][method]
            generated, trace, protocol, matched = generate_method(
                backend, packed, method, mask, budget, max_new_tokens,
                snapkv_window_size, snapkv_kernel_size,
            )
            prediction = generated.generated_answer or ""
            store.write_json(namespace, sample.sample_id, {
                "benchmark": "LongBench-E",
                "task": "2wikimqa_e",
                "sample_id": sample.sample_id,
                "method": method,
                "selected_blocks": mask,
                "budget_blocks": None if method == "full" else budget,
                "block_count": len(sample.blocks),
                "document_tokens": sample.metadata["document_tokens"],
                "official_length_words": sample.metadata["official_length_words"],
                "references": sample.metadata["references"],
                "qa_metrics": qa_metrics(prediction, sample.metadata["references"]),
                "generation_result": generated.to_dict(),
                "candidate_pool_size": None if method == "full" else len(plan["candidate_pool"]),
                "trace": trace,
                "cache_protocol": protocol,
                "matched_target_kv_tokens": matched,
            })
            completed += 1
            print(
                f"[stage-i] {sample_index}/{len(samples)} {method}: "
                f"F1={qa_metrics(prediction, sample.metadata['references'])['answer_f1']:.3f}",
                flush=True,
            )
        store.write_status(
            stage="stage_i_longbench", state="RUNNING",
            completed=completed, skipped=skipped, total=total,
        )
        if sample_index % int(config.get("bundle_every_samples", 2)) == 0:
            store.bundle()
    store.write_status(
        stage="stage_i_longbench", state="COMPLETE",
        completed=completed, skipped=skipped, total=total,
    )
    print(f"[stage-i] COMPLETE: {store.bundle()}", flush=True)


if __name__ == "__main__":
    main()
