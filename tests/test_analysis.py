from pathlib import Path

from bikv.analysis import build_report
from bikv.config import ProjectConfig
from bikv.data import generate_dataset
from bikv.schema import write_jsonl
from bikv.storage import ResultStore


def test_report_exposes_necessity_metrics_without_method_results(tmp_path: Path):
    samples = generate_dataset(n_stories=6, n_blocks=10, seed=5)
    data_path = tmp_path / "data.jsonl"
    write_jsonl(data_path, (sample.to_dict() for sample in samples))
    values = {
        "run_id": "report-test",
        "seed": 5,
        "model_name": "unused",
        "n_stories": 6,
        "n_blocks": 10,
        "block_size": 96,
        "budget_blocks": 3,
        "data_path": str(data_path),
        "output_root": str(tmp_path / "runs"),
        "necessity_min_family_pass_rate": 0.6,
    }
    config = ProjectConfig(tmp_path / "config.json", values)
    store = ResultStore(values["output_root"], values["run_id"])
    for sample in samples:
        store.write_json(
            "necessity",
            sample.sample_id,
            {
                "story_id": sample.story_id,
                "passed": True,
                "symbolic_passed": True,
                "full_generation": {"is_correct": True},
                "ablation_generations": {
                    "without_left": {"is_correct": False},
                    "without_right": {"is_correct": False},
                    "without_both": {"is_correct": False},
                },
                "logprob_drops": {
                    "without_left": 0.2,
                    "without_right": 0.3,
                    "without_both": 0.4,
                },
            },
        )
    report = build_report(config)
    assert report["necessity"]["family_pass_rate"] == 1.0
    assert report["necessity"]["individual_passed_stories"] == 2
    assert report["necessity"]["individual_total_stories"] == 2
    assert report["necessity"]["complete_passed_families"] == 1
    assert report["necessity"]["total_families"] == 1
    assert report["necessity"]["family_eligible_stories"] == 2
    assert report["necessity"]["symbolic_pass_rate"] == 1.0
    assert report["necessity"]["ablation_accuracy"]["without_left"] == 0.0
    assert report["gates"]["necessity_signal"] is True
    assert report["decision"] == "STOP_OR_REDESIGN"


def test_report_distinguishes_individual_passes_from_complete_family_eligibility(tmp_path: Path):
    samples = generate_dataset(n_stories=6, n_blocks=10, seed=17)
    data_path = tmp_path / "data.jsonl"
    write_jsonl(data_path, (sample.to_dict() for sample in samples))
    values = {
        "run_id": "partial-family-report",
        "seed": 17,
        "model_name": "unused",
        "n_stories": 6,
        "n_blocks": 10,
        "block_size": 96,
        "budget_blocks": 3,
        "data_path": str(data_path),
        "output_root": str(tmp_path / "runs"),
        "evaluation_variants": ["neutral"],
        "necessity_min_family_pass_rate": 0.6,
    }
    config = ProjectConfig(tmp_path / "config.json", values)
    store = ResultStore(values["output_root"], values["run_id"])
    test_samples = [sample for sample in samples if sample.split == "test" and sample.variant == "neutral"]
    for index, sample in enumerate(test_samples):
        passed = index == 0
        store.write_json(
            "necessity",
            sample.sample_id,
            {
                "story_id": sample.story_id,
                "passed": passed,
                "symbolic_passed": True,
                "full_generation": {"is_correct": passed},
                "ablation_generations": {
                    "without_left": {"is_correct": False},
                    "without_right": {"is_correct": False},
                    "without_both": {"is_correct": False},
                },
                "logprob_drops": {
                    "without_left": 0.2,
                    "without_right": 0.3,
                    "without_both": 0.4,
                },
            },
        )
    report = build_report(config)
    assert report["necessity"]["individual_passed_stories"] == 1
    assert report["necessity"]["individual_total_stories"] == 2
    assert report["necessity"]["complete_passed_families"] == 0
    assert report["necessity"]["family_eligible_stories"] == 0
