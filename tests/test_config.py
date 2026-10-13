import json

from bikv.config import ProjectConfig


def test_operational_paths_do_not_change_scientific_hash(tmp_path):
    base = {
        "run_id": "x",
        "seed": 1,
        "model_name": "model",
        "n_stories": 3,
        "n_blocks": 6,
        "block_size": 64,
        "budget_blocks": 3,
        "data_path": "a.jsonl",
        "output_root": "runs-a",
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(base), encoding="utf-8")
    base.update({"data_path": "/content/data.jsonl", "output_root": "/content/runs", "mirror_root": "/drive"})
    second.write_text(json.dumps(base), encoding="utf-8")
    assert ProjectConfig.load(first).config_hash == ProjectConfig.load(second).config_hash
