from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from bikv.config import ProjectConfig
from bikv.data import generate_dataset
from bikv.predictor import LinearInteractionModel, train_linear_predictor
from bikv.schema import StorySample, load_samples, write_jsonl
from bikv.storage import ResultStore, stable_mask_key

if TYPE_CHECKING:
    from bikv.model import KVModelBackend


def make_store(config: ProjectConfig) -> ResultStore:
    mirror = config.get("mirror_root")
    return ResultStore(
        config.resolve("output_root"),
        str(config.get("run_id")),
        Path(mirror).expanduser() if mirror else None,
    )


def prepare_dataset(config: ProjectConfig) -> Path:
    samples = generate_dataset(
        n_stories=int(config.get("n_stories")),
        n_blocks=int(config.get("n_blocks")),
        seed=int(config.get("seed")),
    )
    path = config.resolve("data_path")
    write_jsonl(path, (sample.to_dict() for sample in samples))
    summary = {
        "samples": len(samples),
        "stories": len({sample.story_id for sample in samples}),
        "splits": {split: sum(sample.split == split for sample in samples) for split in ("train", "validation", "test")},
        "variants": {variant: sum(sample.variant == variant for sample in samples) for variant in ("neutral", "targeted", "sham")},
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    return path


def _hardware_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for package in ("transformers", "kvpress", "scikit-learn"):
        try:
            manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest[package] = None
    if torch.cuda.is_available():
        manifest.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return manifest


def initialize_run(config: ProjectConfig, store: ResultStore) -> None:
    manifest = {**config.as_manifest(), "environment": _hardware_manifest(), "created_at": time.time()}
    if store.exists("", "manifest"):
        existing = store.read_json("", "manifest")
        if existing.get("config_hash") != config.config_hash:
            raise ValueError(
                "Existing run_id uses a different scientific config. Change run_id instead of mixing results."
            )
    else:
        store.write_json("", "manifest", manifest)


def load_backend(config: ProjectConfig) -> KVModelBackend:
    from bikv.model import KVModelBackend

    torch.manual_seed(int(config.get("seed")))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.get("seed")))
    return KVModelBackend.load(
        model_name=str(config.get("model_name")),
        block_size=int(config.get("block_size")),
        attention_implementation=str(config.get("attention_implementation", "sdpa")),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_chat_template=bool(config.get("use_chat_template", True)),
        prefill_chunk_size=int(config.get("prefill_chunk_size", 2048)),
        prefill_chunk_threshold=int(config.get("prefill_chunk_threshold", 4096)),
    )


def _within_time_budget(started: float, config: ProjectConfig) -> bool:
    minutes = float(config.get("max_gpu_minutes", 0))
    if minutes <= 0:
        return True
    reserve = float(config.get("shutdown_reserve_minutes", 5))
    return (time.monotonic() - started) / 60.0 < max(0.0, minutes - reserve)


def _candidate_boundaries(sample: StorySample, unary: list[float], maximum: int) -> list[int]:
    del sample
    total = len(unary) - 1
    if maximum < total:
        raise ValueError(
            "Budget-matched pair ranking requires every adjacent boundary; "
            f"max_edges_per_sample={maximum} is smaller than {total}"
        )
    return list(range(total))


def _qualification_thresholds(config: ProjectConfig) -> dict[str, float]:
    return {
        "min_candidate_accuracy": float(config.get("qualification_min_candidate_accuracy", 0.90)),
        "min_greedy_accuracy": float(config.get("qualification_min_greedy_accuracy", 0.75)),
        "min_variant_accuracy": float(config.get("qualification_min_variant_accuracy", 0.80)),
        "min_hop_accuracy": float(config.get("qualification_min_hop_accuracy", 0.90)),
        "min_chain_accuracy": float(config.get("qualification_min_chain_accuracy", 0.85)),
        "min_story_stability": float(config.get("qualification_min_story_stability", 0.75)),
        "min_greedy_story_stability": float(config.get("qualification_min_greedy_story_stability", 0.60)),
        "min_counterfactual_pair_rate": float(config.get("qualification_min_counterfactual_pair_rate", 0.75)),
        "min_necessity_rate": float(config.get("qualification_min_necessity_rate", 0.30)),
        "min_position_accuracy": float(config.get("qualification_min_position_accuracy", 0.70)),
    }


def _position_bucket(sample: StorySample) -> str:
    position = int(sample.supporting_blocks[0])
    fraction = position / max(1, len(sample.blocks) - 2)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _candidate_score_mask(backend, packed, candidates: list[str], selected: list[int] | None) -> dict[str, Any]:
    press = None
    if selected is not None:
        from bikv.presses import FixedBlockPress

        press = FixedBlockPress(layout=packed.layout, selected_blocks=selected)
    return backend.score_candidates(packed, candidates, press=press)


def _pack_text_ablation(backend, sample: StorySample, excluded_blocks: set[int]):
    """Remove evidence before tokenization so later hidden states cannot retain it."""
    blocks = [
        "Redacted record. No assignment or verification information is available in this entry."
        if index in excluded_blocks
        else block
        for index, block in enumerate(sample.blocks)
    ]
    return backend.pack(replace(sample, blocks=blocks))


def qualify_backbone(config: ProjectConfig, backend: KVModelBackend | None = None) -> dict[str, Any]:
    from bikv.qualification import render_qualification_report, summarize_qualification

    store = make_store(config)
    initialize_run(config, store)
    allowed_splits = set(config.get("qualification_splits", ["train", "validation"]))
    samples = [
        sample for sample in load_samples(config.resolve("data_path")) if sample.split in allowed_splits
    ]
    if not samples:
        raise ValueError(f"No samples belong to qualification_splits={sorted(allowed_splits)}")
    if backend is None:
        backend = load_backend(config)
    started = time.monotonic()
    completed = 0
    skipped = 0

    for sample in samples:
        if not _within_time_budget(started, config):
            store.write_status(stage="qualify", state="PAUSED_TIME_BUDGET", completed=completed, skipped=skipped)
            store.bundle()
            return {"decision": "INCOMPLETE", "completed": completed, "skipped": skipped}
        if store.exists("qualification/samples", sample.sample_id):
            skipped += 1
            continue

        candidates = [str(value) for value in sample.metadata["candidate_codes"]]
        packed = backend.pack(sample)
        full_candidates = backend.score_candidates(packed, candidates)
        full_greedy = backend.generate(
            packed, max_new_tokens=int(config.get("max_new_tokens", 8))
        ).to_dict()
        payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "split": sample.split,
            "variant": sample.variant,
            "counterfactual_group": sample.metadata["counterfactual_group"],
            "supporting_blocks": sample.supporting_blocks,
            "support_position_bucket": _position_bucket(sample),
            "full_candidates": full_candidates,
            "full_greedy": full_greedy,
            "config_hash": config.config_hash,
        }

        if sample.variant == "neutral":
            left, right = sample.supporting_blocks
            exclusions = {
                "without_left": {left},
                "without_right": {right},
                "without_both": {left, right},
            }
            payload["ablations"] = {
                name: backend.score_candidates(_pack_text_ablation(backend, sample, excluded), candidates)
                for name, excluded in exclusions.items()
            }
            payload["ablation_mode"] = "text_pre_prefill"

            target_lab = str(sample.metadata["target_lab"])
            target_agent = str(sample.metadata["target_agent"])
            agents = [str(value) for value in sample.metadata["candidate_agents"]]
            hop1_question = f"Which technician is assigned to laboratory {target_lab}?"
            hop1_packed = backend.pack(
                sample,
                question=hop1_question,
                answer=target_agent,
                answer_aliases=[target_agent],
            )
            hop1 = backend.score_candidates(hop1_packed, agents)
            payload["hop1_agent"] = hop1

            hop2_question = (
                f"Technician {target_agent} is assigned to laboratory {target_lab}. "
                f"What verification code is assigned to technician {target_agent}?"
            )
            hop2_packed = backend.pack(sample, question=hop2_question)
            payload["hop2_code_given_gold_agent"] = backend.score_candidates(hop2_packed, candidates)

            predicted_agent = str(hop1["prediction"])
            chain_question = (
                f"The technician selected for laboratory {target_lab} is {predicted_agent}. "
                f"What verification code is assigned to technician {predicted_agent}?"
            )
            chain_packed = backend.pack(sample, question=chain_question)
            payload["chain_end_to_end"] = backend.score_candidates(chain_packed, candidates)

        store.write_json("qualification/samples", sample.sample_id, payload)
        completed += 1
        store.write_status(
            stage="qualify",
            state="RUNNING",
            completed=completed,
            skipped=skipped,
            last_sample=sample.sample_id,
            elapsed_minutes=(time.monotonic() - started) / 60.0,
        )
        if completed % max(1, int(config.get("bundle_every_samples", 1))) == 0:
            store.bundle()

    rows = [store.read_json("qualification/samples", sample.sample_id) for sample in samples]
    summary = summarize_qualification(rows, _qualification_thresholds(config))
    summary["config_hash"] = config.config_hash
    store.write_json("qualification", "summary", summary)
    store.write_text("qualification", "report", render_qualification_report(summary))
    store.write_status(stage="qualify", state="COMPLETE", decision=summary["decision"])
    store.bundle()
    return summary


def _score_mask(
    backend: KVModelBackend,
    packed,
    selected: list[int],
    sample: StorySample,
    config: ProjectConfig,
    store: ResultStore,
) -> dict[str, Any]:
    from bikv.presses import FixedBlockPress

    mask_key = stable_mask_key(selected)
    namespace = f"masks/{sample.sample_id}"
    if store.exists(namespace, mask_key):
        return store.read_json(namespace, mask_key)
    press = FixedBlockPress(layout=packed.layout, selected_blocks=selected)
    result = backend.score(packed, press=press)
    payload = {
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "selected_blocks": sorted(selected),
        "config_hash": config.config_hash,
        "result": result.to_dict(),
        "trace": press.trace,
    }
    store.write_json(namespace, mask_key, payload)
    if bool(config.get("bundle_after_each_mask", False)):
        store.bundle()
    return payload


def _score_candidate_mask(
    backend: KVModelBackend,
    packed,
    selected: list[int],
    candidates: list[str],
    sample: StorySample,
    config: ProjectConfig,
    store: ResultStore,
) -> dict[str, Any]:
    from bikv.presses import FixedBlockPress

    mask_key = stable_mask_key(selected)
    namespace = f"proposal_masks/{sample.sample_id}"
    if store.exists(namespace, mask_key):
        return store.read_json(namespace, mask_key)
    press = FixedBlockPress(layout=packed.layout, selected_blocks=selected)
    result = backend.score_candidates(packed, candidates, press=press)
    payload = {
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "selected_blocks": sorted(selected),
        "config_hash": config.config_hash,
        "result": result,
        "trace": press.trace,
    }
    store.write_json(namespace, mask_key, payload)
    if bool(config.get("bundle_after_each_mask", False)):
        store.bundle()
    return payload


def interaction_contrast(utilities: dict[str, float]) -> dict[str, float]:
    """Return the factorial 2x2 effects for an exact-budget intervention."""
    required = {"both", "left_only", "right_only", "neither"}
    if set(utilities) != required:
        raise ValueError(f"Interaction utilities must contain exactly {sorted(required)}")
    both = float(utilities["both"])
    left = float(utilities["left_only"])
    right = float(utilities["right_only"])
    neither = float(utilities["neither"])
    return {
        "interaction": both - left - right + neither,
        "joint_gain": both - neither,
        "left_main_effect": 0.5 * ((both - right) + (left - neither)),
        "right_main_effect": 0.5 * ((both - left) + (right - neither)),
        "left_conditional_on_right": both - right,
        "right_conditional_on_left": both - left,
    }


def necessity_passes(
    full_correct: bool,
    ablation_correct: dict[str, bool],
    logprob_drops: dict[str, float],
    minimum_drop: float,
    require_ablation_failure: bool,
) -> bool:
    required = {"without_left", "without_right", "without_both"}
    if set(ablation_correct) != required or set(logprob_drops) != required:
        raise ValueError("Necessity gate requires all three support ablations")
    probability_gate = all(float(logprob_drops[name]) >= minimum_drop for name in required)
    answer_gate = not require_ablation_failure or all(not bool(ablation_correct[name]) for name in required)
    return bool(full_correct and probability_gate and answer_gate)


def _generate_mask(backend: KVModelBackend, packed, selected: list[int] | None, config: ProjectConfig) -> dict[str, Any]:
    press = None
    if selected is not None:
        from bikv.presses import FixedBlockPress

        press = FixedBlockPress(layout=packed.layout, selected_blocks=selected)
    return backend.generate(
        packed,
        press=press,
        max_new_tokens=int(config.get("max_new_tokens", 8)),
    ).to_dict()


def _necessity_check(
    backend: KVModelBackend,
    packed,
    sample: StorySample,
    config: ProjectConfig,
    store: ResultStore,
    full_score: dict[str, Any],
) -> dict[str, Any]:
    if store.exists("necessity", sample.sample_id):
        return store.read_json("necessity", sample.sample_id)

    left, right = sample.supporting_blocks
    all_blocks = list(range(len(sample.blocks)))
    masks = {
        "without_left": [index for index in all_blocks if index != left],
        "without_right": [index for index in all_blocks if index != right],
        "without_both": [index for index in all_blocks if index not in {left, right}],
    }
    exclusions = {
        "without_left": {left},
        "without_right": {right},
        "without_both": {left, right},
    }
    ablation_packed = {
        name: _pack_text_ablation(backend, sample, excluded)
        for name, excluded in exclusions.items()
    }
    ablation_scores = {
        name: backend.score(ablated).to_dict()
        for name, ablated in ablation_packed.items()
    }
    full_generation = _generate_mask(backend, packed, None, config)
    ablation_generations = {
        name: backend.generate(
            ablated, max_new_tokens=int(config.get("max_new_tokens", 8))
        ).to_dict()
        for name, ablated in ablation_packed.items()
    }
    full_logprob = float(full_score["gold_mean_logprob"])
    drops = {
        name: full_logprob - float(result["gold_mean_logprob"])
        for name, result in ablation_scores.items()
    }
    ablation_correct = {
        name: bool(result["is_correct"]) for name, result in ablation_generations.items()
    }
    symbolic_candidates = sample.metadata.get("symbolic_ablation_candidates", {})
    symbolic_candidate_counts = {
        name: len(set(symbolic_candidates.get(name, [])))
        for name in ("without_left", "without_right", "without_both")
    }
    symbolic_minimum = int(config.get("symbolic_min_ablation_candidates", 3))
    symbolic_passed = all(
        count >= symbolic_minimum for count in symbolic_candidate_counts.values()
    )
    minimum_drop = float(config.get("necessity_min_logprob_drop", 0.2))
    require_failure = bool(config.get("necessity_require_ablation_failure", True))
    model_passed = necessity_passes(
        full_correct=bool(full_generation["is_correct"]),
        ablation_correct=ablation_correct,
        logprob_drops=drops,
        minimum_drop=minimum_drop,
        require_ablation_failure=require_failure,
    )
    passed = bool(symbolic_passed and model_passed)
    payload = {
        "sample_id": sample.sample_id,
        "story_id": sample.story_id,
        "split": sample.split,
        "variant": sample.variant,
        "counterfactual_group": sample.metadata["counterfactual_group"],
        "supporting_blocks": sample.supporting_blocks,
        "masks": masks,
        "ablation_mode": "text_pre_prefill",
        "minimum_logprob_drop": minimum_drop,
        "symbolic_minimum_candidates": symbolic_minimum,
        "symbolic_candidate_counts": symbolic_candidate_counts,
        "symbolic_passed": symbolic_passed,
        "model_passed": model_passed,
        "require_ablation_failure": require_failure,
        "full_score": full_score,
        "ablation_scores": ablation_scores,
        "full_generation": full_generation,
        "ablation_generations": ablation_generations,
        "logprob_drops": drops,
        "passed": passed,
        "config_hash": config.config_hash,
    }
    store.write_json("necessity", sample.sample_id, payload)
    return payload


def eligible_story_ids(store: ResultStore, samples: list[StorySample]) -> set[str]:
    return eligible_story_ids_for_variants(store, samples, {sample.variant for sample in samples})


def eligible_story_ids_for_variants(
    store: ResultStore, samples: list[StorySample], allowed_variants: set[str]
) -> set[str]:
    families: dict[str, list[StorySample]] = {}
    for sample in samples:
        if sample.variant not in allowed_variants:
            continue
        family = str(sample.metadata.get("counterfactual_group", sample.story_id))
        families.setdefault(family, []).append(sample)
    eligible: set[str] = set()
    for family_samples in families.values():
        expected = {sample.sample_id for sample in family_samples}
        passed = {
            sample.sample_id
            for sample in family_samples
            if store.exists("necessity", sample.sample_id)
            and bool(store.read_json("necessity", sample.sample_id).get("passed"))
        }
        if passed == expected:
            eligible.update(sample.story_id for sample in family_samples)
    return eligible


def label_interactions(config: ProjectConfig, backend: KVModelBackend | None = None) -> dict[str, int]:
    from bikv.presses import (
        FeatureCapturePress,
        aggregate_layer_features,
        matched_interaction_masks,
        pair_proposal,
    )

    store = make_store(config)
    initialize_run(config, store)
    if bool(config.get("require_qualification_go", False)):
        if not store.exists("qualification", "summary"):
            raise RuntimeError("Run the qualify stage before label; no qualification summary exists")
        decision = store.read_json("qualification", "summary").get("decision")
        if decision != "GO_LABEL":
            raise RuntimeError(f"Backbone qualification decision is {decision}, not GO_LABEL")
    label_splits = set(config.get("label_splits", ["train", "validation"]))
    label_variants = set(config.get("label_variants", ["neutral"]))
    samples = [
        sample
        for sample in load_samples(config.resolve("data_path"))
        if sample.split in label_splits and sample.variant in label_variants
    ]
    total_samples = len(samples)
    existing_labels = sum(store.exists("labels", sample.sample_id) for sample in samples)
    remaining_at_start = total_samples - existing_labels
    print(
        f"[label] Starting/resuming: total={total_samples}, already_done={existing_labels}, "
        f"remaining={remaining_at_start}",
        flush=True,
    )
    if backend is None:
        print(f"[label] Loading model {config.get('model_name')}...", flush=True)
        backend = load_backend(config)
        print("[label] Model loaded.", flush=True)
    started = time.monotonic()
    completed = 0
    skipped = 0
    bundle_every = max(1, int(config.get("bundle_every_samples", 1)))

    for sample_index, sample in enumerate(samples, start=1):
        if not _within_time_budget(started, config):
            store.write_status(stage="label", state="PAUSED_TIME_BUDGET", completed=completed, skipped=skipped)
            print(
                f"[label] Paused before sample {sample_index}/{total_samples}: GPU time budget reached. "
                f"newly_done={completed}, skipped_existing={skipped}",
                flush=True,
            )
            break
        if store.exists("labels", sample.sample_id):
            skipped += 1
            print(
                f"[label] Sample {sample_index}/{total_samples} {sample.sample_id}: already complete, skipped "
                f"({skipped}/{existing_labels} restored)",
                flush=True,
            )
            continue
        sample_started = time.monotonic()
        print(
            f"[label] Sample {sample_index}/{total_samples} {sample.sample_id} "
            f"({sample.split}/{sample.variant}): capture + necessity...",
            flush=True,
        )
        packed = backend.pack(sample)
        capture_suffix = ".json.gz"
        if store.exists("capture", sample.sample_id, capture_suffix):
            capture_payload = store.read_json("capture", sample.sample_id, capture_suffix)
        else:
            full_result = backend.score(packed)
            capture_press = FeatureCapturePress(layout=packed.layout)
            layers = backend.capture_query_aware_features(packed, capture_press)
            capture_payload = {
                "sample_id": sample.sample_id,
                "story_id": sample.story_id,
                "split": sample.split,
                "variant": sample.variant,
                "config_hash": config.config_hash,
                "layout": packed.layout.to_dict(),
                "content_tokens": int(sum(packed.layout.block_content_lengths)),
                "physical_document_tokens": int(packed.layout.n_blocks * packed.layout.block_size),
                "content_fraction": float(
                    sum(packed.layout.block_content_lengths)
                    / (packed.layout.n_blocks * packed.layout.block_size)
                ),
                "full_result": full_result.to_dict(),
                "layers": layers,
                "protocol": "disposable_query_diagnostic_then_clean_document_prefill",
            }
            store.write_json("capture", sample.sample_id, capture_payload, compress=True)

        edge_labels = []
        necessity = _necessity_check(
            backend, packed, sample, config, store, capture_payload["full_result"]
        )
        if necessity["passed"]:
            layers = sorted(capture_payload["layers"], key=lambda row: row["layer_index"])
            if not layers:
                raise RuntimeError(f"No layer features captured for {sample.sample_id}")
            unary, aggregated_edges = aggregate_layer_features(
                layers, skip_first_layers=int(config.get("feature_skip_first_layers", 2))
            )
            max_edges = int(config.get("max_edges_per_sample", len(unary) - 1))
            boundaries = _candidate_boundaries(sample, unary, max_edges)
            budget_blocks = int(config.get("budget_blocks"))
            candidates = [str(value) for value in sample.metadata["candidate_codes"]]
            singleton_mask = sorted(range(len(unary)), key=lambda index: unary[index], reverse=True)[:budget_blocks]
            singleton_payload = _score_candidate_mask(
                backend, packed, singleton_mask, candidates, sample, config, store
            )
            singleton_utility = float(singleton_payload["result"]["gold_margin"])
            print(
                f"[label] Sample {sample_index}/{total_samples}: necessity PASS; "
                f"labeling {len(boundaries)} matched 2x2 boundary interventions",
                flush=True,
            )

            for boundary_index, boundary in enumerate(boundaries, start=1):
                intervention_masks = matched_interaction_masks(unary, boundary, budget_blocks)
                print(
                    f"[label] Sample {sample_index}/{total_samples}; interaction "
                    f"{boundary_index}/{len(boundaries)} (boundary={boundary}, "
                    f"4 matched masks; cached duplicates are reused)",
                    flush=True,
                )
                intervention_payloads = {
                    name: _score_candidate_mask(
                        backend, packed, mask, candidates, sample, config, store
                    )
                    for name, mask in intervention_masks.items()
                }
                utilities = {
                    name: float(payload["result"]["gold_margin"])
                    for name, payload in intervention_payloads.items()
                }
                effects = interaction_contrast(utilities)
                deployment_mask = pair_proposal(unary, boundary, budget_blocks)
                deployment_payload = _score_candidate_mask(
                    backend, packed, deployment_mask, candidates, sample, config, store
                )
                deployment_utility = float(deployment_payload["result"]["gold_margin"])
                edge_labels.append(
                    {
                        "boundary": boundary,
                        "is_support_boundary": boundary == int(sample.metadata["support_boundary"]),
                        "intervention_masks": intervention_masks,
                        "intervention_utilities": utilities,
                        "intervention_effects": effects,
                        "intervention_signature": "|".join(
                            ",".join(str(value) for value in intervention_masks[name])
                            for name in ("both", "left_only", "right_only", "neither")
                        ) + "|deploy=" + ",".join(str(value) for value in deployment_mask),
                        "proposal_mask": deployment_mask,
                        "proposal_utility": deployment_utility,
                        "proposal_result": deployment_payload["result"],
                        "singleton_mask": singleton_mask,
                        "singleton_utility": singleton_utility,
                        "aggregated_features": aggregated_edges[boundary],
                    }
                )
        else:
            print(
                f"[label] Sample {sample_index}/{total_samples}: necessity FAIL; no edge masks computed",
                flush=True,
            )

        label_payload = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "split": sample.split,
            "variant": sample.variant,
            "counterfactual_group": sample.metadata["counterfactual_group"],
            "supporting_blocks": sample.supporting_blocks,
            "evidence_order": sample.metadata.get("evidence_order"),
            "manipulated_boundary": sample.manipulated_boundary,
            "config_hash": config.config_hash,
            "full_result": capture_payload["full_result"],
            "necessity_passed": bool(necessity["passed"]),
            "edge_labels": edge_labels,
        }
        store.write_json("labels", sample.sample_id, label_payload)
        completed += 1
        store.write_status(
            stage="label",
            state="RUNNING",
            completed=completed,
            skipped=skipped,
            last_sample=sample.sample_id,
            elapsed_minutes=(time.monotonic() - started) / 60.0,
        )
        if completed % bundle_every == 0:
            store.bundle()
        elapsed = time.monotonic() - started
        average = elapsed / completed
        unfinished = max(0, remaining_at_start - completed)
        eta = average * unfinished
        sample_seconds = time.monotonic() - sample_started
        print(
            f"[label] Completed sample {sample_index}/{total_samples} in {sample_seconds:.1f}s; "
            f"newly_done={completed}/{remaining_at_start}, skipped_existing={skipped}, "
            f"elapsed={elapsed / 60.0:.1f}m, ETA~{eta / 60.0:.1f}m",
            flush=True,
        )
    else:
        store.write_status(stage="label", state="COMPLETE", completed=completed, skipped=skipped)
        store.bundle()
        print(
            f"[label] COMPLETE: newly_done={completed}, skipped_existing={skipped}, "
            f"total={total_samples}, elapsed={(time.monotonic() - started) / 60.0:.1f}m",
            flush=True,
        )
    return {"completed": completed, "skipped": skipped}


def train_predictor(config: ProjectConfig) -> dict[str, Any]:
    store = make_store(config)
    initialize_run(config, store)
    samples = {sample.sample_id: sample for sample in load_samples(config.resolve("data_path"))}
    label_variants = set(config.get("label_variants", ["neutral"]))
    eligible = eligible_story_ids_for_variants(store, list(samples.values()), label_variants)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for sample_id, sample in samples.items():
        if sample.story_id not in eligible:
            continue
        if not store.exists("labels", sample_id):
            continue
        label = store.read_json("labels", sample_id)
        destination = train_rows if sample.split == "train" else validation_rows if sample.split == "validation" else None
        if destination is None:
            continue
        seen_interventions: set[str] = set()
        for edge in label["edge_labels"]:
            signature = str(edge.get("intervention_signature", edge["boundary"]))
            if signature in seen_interventions:
                continue
            seen_interventions.add(signature)
            destination.append(
                {
                    "story_id": sample.story_id,
                    "sample_id": sample.sample_id,
                    "counterfactual_group": sample.metadata["counterfactual_group"],
                    "boundary": edge["boundary"],
                    "features": edge["aggregated_features"],
                    "target": edge["intervention_effects"]["interaction"],
                    "interaction": edge["intervention_effects"]["interaction"],
                    "joint_gain": edge["intervention_effects"]["joint_gain"],
                    "proposal_utility": edge["proposal_utility"],
                    "intervention_signature": signature,
                    "is_support_boundary": edge["is_support_boundary"],
                    "supporting_blocks": sample.supporting_blocks,
                    "proposal_mask": edge["proposal_mask"],
                    "singleton_mask": edge["singleton_mask"],
                    "singleton_utility": edge["singleton_utility"],
                }
            )
    if not train_rows:
        payload = {
            "status": "NO_ELIGIBLE_TRAIN_ROWS",
            "eligible_story_ids": sorted(eligible),
            "config_hash": config.config_hash,
        }
        store.write_status(stage="train", state="NO_ELIGIBLE_TRAIN_ROWS")
        store.write_json("models", "pair_model_unavailable", payload)
        store.bundle()
        return payload
    model = train_linear_predictor(train_rows, validation_rows)
    payload = asdict(model)
    metrics = model.validation_metrics or {}
    payload["validation_gate_passed"] = bool(
        float(metrics.get("validation_support_retention", 0.0))
        >= float(config.get("validation_min_support_retention", 0.50))
        and float(metrics.get("validation_support_retention_lift", 0.0))
        >= float(config.get("validation_min_support_retention_lift", 0.10))
        and float(metrics.get("validation_mean_margin_lift_over_singleton", -1e9))
        >= float(config.get("validation_min_margin_lift", 0.0))
        and float(metrics.get("validation_support_interaction_gap", -1e9))
        >= float(config.get("validation_min_support_interaction_gap", 0.0))
        and float(metrics.get("validation_support_positive_interaction_rate", 0.0))
        >= float(config.get("validation_min_support_positive_rate", 0.60))
        and float(metrics.get("validation_support_interaction_gap_bootstrap_low", -1e9))
        >= float(config.get("validation_min_support_interaction_ci_low", 0.0))
    )
    payload["config_hash"] = config.config_hash
    store.write_json("models", "pair_model", payload)
    store.write_status(stage="train", state="COMPLETE", metrics=model.validation_metrics)
    store.bundle()
    return payload


def recapture_features(config: ProjectConfig, backend: KVModelBackend | None = None) -> dict[str, int]:
    """Capture v11 query/head/layer diagnostics without recomputing factorial labels."""
    from bikv.presses import RichFeatureCapturePress

    store = make_store(config)
    initialize_run(config, store)
    splits = set(config.get("recapture_splits", ["train", "validation"]))
    variants = set(config.get("recapture_variants", ["neutral"]))
    samples = [
        sample
        for sample in load_samples(config.resolve("data_path"))
        if sample.split in splits and sample.variant in variants
    ]
    if backend is None:
        print(f"[recapture] Loading model {config.get('model_name')}...", flush=True)
        backend = load_backend(config)
    started = time.monotonic()
    completed = 0
    skipped = 0
    total = len(samples)
    suffix = ".json.gz"
    for index, sample in enumerate(samples, start=1):
        if store.exists("rich_capture", sample.sample_id, suffix):
            skipped += 1
            print(f"[recapture] {index}/{total} {sample.sample_id}: cached", flush=True)
            continue
        if not _within_time_budget(started, config):
            store.write_status(
                stage="recapture",
                state="STOPPED_TIME_BUDGET",
                completed=completed,
                skipped=skipped,
                total=total,
            )
            store.bundle()
            print("[recapture] Stopped before reserve window; resume with the same command.", flush=True)
            return {"completed": completed, "skipped": skipped, "remaining": total - index + 1}
        print(f"[recapture] {index}/{total} {sample.sample_id}: rich diagnostic pass...", flush=True)
        packed = backend.pack(sample)
        press = RichFeatureCapturePress(layout=packed.layout)
        capture = backend.capture_rich_query_aware_features(packed, press)
        payload = {
            "sample_id": sample.sample_id,
            "story_id": sample.story_id,
            "counterfactual_group": sample.metadata["counterfactual_group"],
            "split": sample.split,
            "variant": sample.variant,
            "evidence_order": sample.metadata.get("evidence_order"),
            "supporting_blocks": sample.supporting_blocks,
            "layout": packed.layout.to_dict(),
            "config_hash": config.config_hash,
            "protocol": "disposable_query_attention_capture_no_scoring_cache_reuse",
            "feature_schema": "v11_head_layer_query_attention_v1",
            **capture,
        }
        store.write_json("rich_capture", sample.sample_id, payload, compress=True)
        completed += 1
        elapsed = time.monotonic() - started
        processed = completed + skipped
        eta = elapsed / max(1, completed) * max(0, total - processed) if completed else 0.0
        store.write_status(
            stage="recapture",
            state="RUNNING",
            completed=completed,
            skipped=skipped,
            total=total,
            last_sample=sample.sample_id,
            elapsed_minutes=elapsed / 60.0,
            eta_minutes=eta / 60.0,
        )
        bundle_every = max(1, int(config.get("bundle_every_samples", 4)))
        if bool(config.get("bundle_after_each_mask", False)) or completed % bundle_every == 0:
            store.bundle()
        print(
            f"[recapture] {index}/{total} done; elapsed={elapsed / 60.0:.1f}m, "
            f"ETA~{eta / 60.0:.1f}m, peak={capture['peak_vram_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )
    store.write_status(
        stage="recapture", state="COMPLETE", completed=completed, skipped=skipped, total=total
    )
    store.bundle()
    print(f"[recapture] COMPLETE: new={completed}, cached={skipped}, total={total}", flush=True)
    return {"completed": completed, "skipped": skipped, "remaining": 0}


def evaluate(config: ProjectConfig, backend: KVModelBackend | None = None) -> dict[str, int]:
    from bikv.presses import FeatureCapturePress, FixedBlockPress, ProtectedKnormPress, select_shared_plan

    store = make_store(config)
    initialize_run(config, store)
    all_samples = load_samples(config.resolve("data_path"))
    evaluation_variants = set(config.get("evaluation_variants", ["neutral"]))
    test_samples = [
        sample
        for sample in all_samples
        if sample.split == "test" and sample.variant in evaluation_variants
    ]
    model_payload = store.read_json("models", "pair_model") if store.exists("models", "pair_model") else None
    if bool(config.get("require_validation_gate", True)) and not bool(
        model_payload and model_payload.get("validation_gate_passed")
    ):
        raise RuntimeError("Validation gate failed; sealed test evaluation is intentionally blocked")
    if backend is None:
        backend = load_backend(config)
    print(f"[evaluate] Running sealed test necessity gate for {len(test_samples)} samples...", flush=True)
    for sample in test_samples:
        if store.exists("necessity", sample.sample_id):
            continue
        packed = backend.pack(sample)
        full_score = backend.score(packed).to_dict()
        _necessity_check(backend, packed, sample, config, store, full_score)
        store.bundle()
    eligible = eligible_story_ids_for_variants(store, all_samples, evaluation_variants)
    samples = [sample for sample in test_samples if sample.story_id in eligible]
    print(
        f"[evaluate] Necessity-eligible test samples: {len(samples)}/{len(test_samples)}",
        flush=True,
    )
    predictor = (
        LinearInteractionModel(
            **{
                key: value
                for key, value in model_payload.items()
                if key not in {"config_hash", "validation_gate_passed"}
            }
        )
        if model_payload and "coefficients" in model_payload
        else None
    )
    methods = list(config.get("evaluation_methods", ["full", "singleton", "neighbor", "learned", "oracle"]))
    started = time.monotonic()
    completed = 0
    skipped = 0

    for sample in samples:
        packed = backend.pack(sample)
        capture_suffix = ".json.gz"
        if store.exists("capture", sample.sample_id, capture_suffix):
            capture_payload = store.read_json("capture", sample.sample_id, capture_suffix)
            layers = capture_payload["layers"]
        else:
            capture_press = FeatureCapturePress(layout=packed.layout)
            layers = backend.capture_query_aware_features(packed, capture_press)
            store.write_json(
                "capture",
                sample.sample_id,
                {
                    "sample_id": sample.sample_id,
                    "story_id": sample.story_id,
                    "split": sample.split,
                    "variant": sample.variant,
                    "config_hash": config.config_hash,
                    "layout": packed.layout.to_dict(),
                    "layers": layers,
                    "protocol": "disposable_query_diagnostic_only",
                },
                compress=True,
            )
        for method in methods:
            if not _within_time_budget(started, config):
                store.write_status(stage="evaluate", state="PAUSED_TIME_BUDGET", completed=completed, skipped=skipped)
                store.bundle()
                return {"completed": completed, "skipped": skipped}
            if store.exists(f"evaluation/{method}", sample.sample_id):
                skipped += 1
                continue
            press = None
            if method == "knorm":
                press = ProtectedKnormPress(
                    layout=packed.layout,
                    retained_document_tokens=int(config.get("budget_blocks")) * packed.layout.block_size,
                    uncompressed_layers=int(config.get("knorm_uncompressed_layers", 0)),
                )
                plan_details = {
                    "mode": "protected_per_head_per_layer_token_knorm",
                    "retained_document_tokens": int(config.get("budget_blocks")) * packed.layout.block_size,
                }
            elif method != "full":
                if method == "learned" and predictor is None:
                    continue
                plan, plan_details = select_shared_plan(
                    layers=layers,
                    budget_blocks=int(config.get("budget_blocks")),
                    mode=method,
                    predictor=predictor if method == "learned" else None,
                    supporting_blocks=sample.supporting_blocks if method in {"oracle", "task_oracle"} else None,
                    continuity_weight=float(config.get("continuity_weight", 1.0)),
                    seed=int(config.get("seed"))
                    + int(hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:8], 16),
                    skip_first_layers=int(config.get("feature_skip_first_layers", 2)),
                )
                press = FixedBlockPress(layout=packed.layout, selected_blocks=plan.selected)
            else:
                plan_details = {"mode": "full"}
            result = backend.generate(packed, press=press, max_new_tokens=int(config.get("max_new_tokens", 8)))
            candidates = [str(value) for value in sample.metadata["candidate_codes"]]
            candidate_result = backend.score_candidates(packed, candidates, press=press)
            trace = press.trace if press is not None else []
            final_selected = (
                trace[-1]["selected_blocks"]
                if trace and "selected_blocks" in trace[-1]
                else list(range(len(sample.blocks))) if method == "full" else []
            )
            if method == "knorm" and trace:
                support_retained = float(
                    sum(trace[-1]["block_retention"][index] for index in sample.supporting_blocks)
                    / len(sample.supporting_blocks)
                )
            else:
                support_retained = float(
                    all(index in final_selected for index in sample.supporting_blocks)
                )
            payload = {
                "sample_id": sample.sample_id,
                "story_id": sample.story_id,
                "split": sample.split,
                "variant": sample.variant,
                "evidence_order": sample.metadata.get("evidence_order"),
                "method": method,
                "supporting_blocks": sample.supporting_blocks,
                "support_retained_final_layer": support_retained,
                "retained_document_blocks": len(final_selected) if method != "knorm" else None,
                "config_hash": config.config_hash,
                "result": result.to_dict(),
                "candidate_result": candidate_result,
                "trace": trace,
                "shared_plan": plan_details,
                "selection_scope": "one_mask_all_layers",
                "cache_protocol": "document_compress_then_query",
            }
            store.write_json(f"evaluation/{method}", sample.sample_id, payload)
            completed += 1
        if int(config.get("bundle_every_samples", 1)) > 0:
            store.bundle()
    store.write_status(stage="evaluate", state="COMPLETE", completed=completed, skipped=skipped)
    store.bundle()
    return {"completed": completed, "skipped": skipped}
