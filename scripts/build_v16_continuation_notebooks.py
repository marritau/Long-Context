from __future__ import annotations

import json
from pathlib import Path


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def common_cells(title: str, work_name: str) -> list[dict]:
    return [
        markdown(
            f"# {title}\n\n"
            "This is one independent frozen shard. It can run in Colab or Kaggle and does not share a run id "
            "with any other shard. Completed rows are mirrored immediately to Google Drive in Colab. In Kaggle, "
            "run with Save Version / Run All; checkpoints and the rolling artifact zip are written to /kaggle/working.\n"
        ),
        code(
            "from pathlib import Path\n"
            "import importlib.util, json, os, shutil, subprocess, sys, zipfile\n"
            "import torch\n"
            "IS_KAGGLE = Path('/kaggle/input').exists() and Path('/kaggle/working').exists()\n"
            "IS_COLAB = Path('/var/colab/hostname').exists() and not IS_KAGGLE\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('Select a GPU accelerator before continuing.')\n"
            "gpu = torch.cuda.get_device_properties(0)\n"
            "print('Platform:', 'Colab' if IS_COLAB else 'Kaggle' if IS_KAGGLE else 'local')\n"
            "print('GPU:', gpu.name, '| VRAM GiB:', round(gpu.total_memory / 2**30, 1))\n"
        ),
        code(
            f"WORK = Path('/content/{work_name}') if IS_COLAB else Path('/kaggle/working/{work_name}')\n"
            "if IS_COLAB:\n"
            "    from google.colab import drive, files\n"
            "    drive.mount('/content/drive')\n"
            "    MIRROR = Path('/content/drive/MyDrive/bikv_runs')\n"
            "    candidates = list(Path('/content').glob('boundary_interaction_kv_v16_continuation_suite*.zip'))\n"
            "    if not candidates:\n"
            "        files.upload()\n"
            "        candidates = list(Path('/content').glob('boundary_interaction_kv_v16_continuation_suite*.zip'))\n"
            "elif IS_KAGGLE:\n"
            "    MIRROR = None\n"
            "    candidates = list(Path('/kaggle/input').rglob('boundary_interaction_kv_v16_continuation_suite*.zip'))\n"
            "    candidates += list(Path('/kaggle/working').glob('boundary_interaction_kv_v16_continuation_suite*.zip'))\n"
            "else:\n"
            "    MIRROR = None\n"
            "    candidates = list(Path.cwd().glob('boundary_interaction_kv_v16_continuation_suite*.zip'))\n"
            "SOURCE = WORK / 'source_v16'\n"
            "if SOURCE.exists():\n"
            "    shutil.rmtree(SOURCE)\n"
            "SOURCE.mkdir(parents=True, exist_ok=True)\n"
            "if candidates:\n"
            "    archive_path = max(candidates, key=lambda path: path.stat().st_mtime)\n"
            "    with zipfile.ZipFile(archive_path) as archive:\n"
            "        archive.extractall(SOURCE)\n"
            "elif IS_KAGGLE:\n"
            "    input_projects = [p.parent for p in Path('/kaggle/input').rglob('pyproject.toml') if (p.parent / 'src' / 'bikv').is_dir()]\n"
            "    if len(input_projects) != 1:\n"
            "        raise FileNotFoundError(f'Expected one extracted BIKV Dataset project, found {input_projects}')\n"
            "    shutil.copytree(input_projects[0], SOURCE / 'boundary_interaction_kv', dirs_exist_ok=True)\n"
            "else:\n"
            "    raise FileNotFoundError('Provide boundary_interaction_kv_v16_continuation_suite.zip.')\n"
            "projects = list(SOURCE.rglob('pyproject.toml'))\n"
            "if len(projects) != 1:\n"
            "    raise RuntimeError(f'Expected one project, found {len(projects)}')\n"
            "PROJECT = projects[0].parent\n"
            "print('Platform:', 'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'local', '| Project:', PROJECT)\n"
        ),
        code(
            "os.environ['HF_HOME'] = str(WORK / 'hf_cache')\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT), 'nltk>=3.9,<4'], check=True)\n"
            "if sys.version_info >= (3, 13):\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'fire==0.7.1'], check=True)\n"
            "env = {**os.environ, 'PYTHONPATH': f\"{PROJECT / 'src'}:{PROJECT / 'scripts'}\"}\n"
            "def stream(command):\n"
            "    print(' '.join(map(str, command)), flush=True)\n"
            "    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)\n"
            "    for line in process.stdout:\n"
            "        print(line, end='', flush=True)\n"
            "    return_code = process.wait()\n"
            "    if return_code:\n"
            "        raise subprocess.CalledProcessError(return_code, command)\n"
            "print('Installed frozen BIKV continuation suite.')\n"
        ),
    ]


def final_cells() -> list[dict]:
    return [
        markdown(
            "## Run or resume\n\n"
            "Re-run this cell after a Colab disconnect. Existing rows and captures are restored from Drive and skipped. "
            "A time-budget pause is normal: reconnect and run the same cell again.\n"
        ),
        code(
            "stream([sys.executable, '-u', str(PROJECT / 'scripts' / 'stage_g1b_g2.py'), '--config', str(runtime_path)])\n"
        ),
        code(
            "run_root = Path(base['output_root']) / base['run_id']\n"
            "stream([sys.executable, str(PROJECT / 'scripts' / 'analyze_stage_g1b_g2.py'), str(run_root), '--seed', str(base['seed'])])\n"
            "from bikv.storage import ResultStore\n"
            "store = ResultStore(base['output_root'], base['run_id'], base.get('mirror_root'))\n"
            "bundle = store.bundle()\n"
            "print('STATUS:', (run_root / 'status.json').read_text())\n"
            "print('REPORT:', (run_root / 'analysis' / 'report.md').read_text())\n"
            "print('Artifact:', bundle)\n"
            "if IS_COLAB:\n"
            "    from google.colab import files\n"
            "    files.download(str(bundle))\n"
            "else:\n"
            "    print('Kaggle output is in /kaggle/working. Save the notebook version to retain it.')\n"
        ),
    ]


def g1b_notebook(label: str, blocks: int) -> dict:
    cells = common_cells(f"BIKV v16 G1b {label} confirmation/rescue", f"bikv_v16_g1b_{label}_work")
    cells.extend([
        markdown(
            "## Frozen experiment\n\n"
            "Runs 64 held-out stories from 32 complete counterfactual families. The fixed-memory arm uses B=6. "
            "The fixed-ratio arm uses the preregistered 3/7 retention. A failed full-context gate is retained and "
            "reported, but methods still run under the explicit exploratory-rescue label. No selector is retrained.\n"
        ),
        code(
            f"BLOCKS = {blocks}\n"
            "base = json.loads((PROJECT / 'configs' / 'v15_g1b_confirmation.json').read_text())\n"
            f"base['run_id'] = 'bikv-v16-g1b-{label}-confirmation-rescue-seed-20261018'\n"
            "dataset_root = PROJECT / 'data' / 'v15_g1b_confirmation'\n"
            "dataset = dataset_root / f'g1b_{BLOCKS:03d}_blocks.jsonl'\n"
            "base['n_blocks'] = BLOCKS\n"
            "base['data_path'] = str(dataset)\n"
            "base['g1b_datasets'] = {str(BLOCKS): str(dataset)}\n"
            "base['g1b_audit_path'] = str(dataset_root / 'g1b_audit.json')\n"
            "base['frozen_models_path'] = str(PROJECT / 'configs' / 'v14_g1b_g2_frozen_models.json')\n"
            "base['g1b_run_failed_gate_as_rescue'] = True\n"
            "base['g1b_qualification_only'] = False\n"
            "base['run_g1b'] = True\n"
            "base['run_g2'] = False\n"
            "base['output_root'] = str(WORK / 'runs')\n"
            "base['mirror_root'] = str(MIRROR) if MIRROR else None\n"
            "runtime_path = WORK / f'runtime_g1b_{BLOCKS:03d}.json'\n"
            "runtime_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "runtime_path.write_text(json.dumps(base, indent=2), encoding='utf-8')\n"
            "print(runtime_path.read_text())\n"
        ),
    ])
    cells.extend(final_cells())
    return notebook_payload(f"BIKV_V16_G1b_{label}_Confirmation_Colab_Kaggle.ipynb", cells)


def ruler_notebook(label: str, length: int) -> dict:
    cells = common_cells(f"BIKV v16 G2 RULER VT {label} comparison/rescue", f"bikv_v16_ruler_{label}_work")
    cells.extend([
        markdown(
            "## Frozen official benchmark\n\n"
            "Uses the pinned NVIDIA RULER Variable Tracking generator at revision c3f5e3b and 50 examples. "
            "At 8k the failed full-context gate remains visible; compressed methods run as an exploratory rescue test. "
            "The method list and frozen selector weights are unchanged.\n"
        ),
        code(
            f"LENGTH = {length}\n"
            f"RULER_DATA = (Path('/content/drive/MyDrive/bikv_datasets/v16_ruler_vt_{label}_seed_20261016') if IS_COLAB "
            "else WORK / 'ruler_data')\n"
            "RULER = WORK / 'RULER'\n"
            "if not RULER.exists():\n"
            "    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/NVIDIA/RULER.git', str(RULER)], check=True)\n"
            "subprocess.run(['git', '-C', str(RULER), 'fetch', 'origin'], check=True)\n"
            "subprocess.run(['git', '-C', str(RULER), 'checkout', '--detach', 'c3f5e3b'], check=True)\n"
            "subprocess.run([sys.executable, str(PROJECT / 'scripts' / 'prepare_ruler_vt.py'), "
            "'--repo', str(RULER), '--output', str(RULER_DATA), '--model', 'Qwen/Qwen2.5-3B-Instruct', "
            "'--revision', 'c3f5e3b', '--lengths', str(LENGTH), '--samples', '50', '--seed', '20261016'], "
            "cwd=PROJECT, env=env, check=True)\n"
            "print((RULER_DATA / 'ruler_manifest.json').read_text())\n"
        ),
        code(
            "base = json.loads((PROJECT / 'configs' / 'v15_g2_ruler_qualification.json').read_text())\n"
            f"base['run_id'] = 'bikv-v16-g2-ruler-vt-{label}-comparison-rescue-seed-20261016'\n"
            "dataset = RULER_DATA / f'vt_{LENGTH}' / 'validation.jsonl'\n"
            "base['data_path'] = str(dataset)\n"
            "base['ruler_manifest_path'] = str(RULER_DATA / 'ruler_manifest.json')\n"
            "base['ruler_datasets'] = {str(LENGTH): str(dataset)}\n"
            "base['frozen_models_path'] = str(PROJECT / 'configs' / 'v14_g1b_g2_frozen_models.json')\n"
            "base['ruler_run_failed_gate_as_rescue'] = True\n"
            "base['ruler_qualification_only'] = False\n"
            "base['run_g1b'] = False\n"
            "base['run_g2'] = True\n"
            "base['output_root'] = str(WORK / 'runs')\n"
            "base['mirror_root'] = str(MIRROR) if MIRROR else None\n"
            "runtime_path = WORK / f'runtime_ruler_{LENGTH}.json'\n"
            "runtime_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "runtime_path.write_text(json.dumps(base, indent=2), encoding='utf-8')\n"
            "print(runtime_path.read_text())\n"
        ),
    ])
    cells.extend(final_cells())
    return notebook_payload(f"BIKV_V16_G2_RULER_{label}_Comparison_Colab_Kaggle.ipynb", cells)


def notebook_payload(name: str, cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output = root / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        ("BIKV_V16_G1b_2k_Confirmation_Colab_Kaggle.ipynb", g1b_notebook("2k", 21)),
        ("BIKV_V16_G1b_4k_Confirmation_Colab_Kaggle.ipynb", g1b_notebook("4k", 42)),
        ("BIKV_V16_G1b_8k_Confirmation_Colab_Kaggle.ipynb", g1b_notebook("8k", 84)),
        ("BIKV_V16_G2_RULER_4k_Comparison_Colab_Kaggle.ipynb", ruler_notebook("4k", 4096)),
        ("BIKV_V16_G2_RULER_8k_Comparison_Colab_Kaggle.ipynb", ruler_notebook("8k", 8192)),
    ]
    for name, payload in payloads:
        path = output / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
