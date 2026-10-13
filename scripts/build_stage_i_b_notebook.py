from __future__ import annotations

import json
from pathlib import Path


ARCHIVE = "boundary_interaction_kv_v19_stage_i_b_longbench_2wiki"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    name = "BIKV_V19_Stage_I_B_LongBench_2Wiki_Replication_Colab_Kaggle.ipynb"
    cells = [
        markdown(
            "# BIKV v19 Stage I-B: LongBench-E 2Wiki replication\n\n"
            "Confirmatory replication on ranks 25-48 of the same frozen eligible cohort. The notebook aborts on "
            "any overlap with Stage I. Models, methods, budget and length filter are unchanged.\n"
        ),
        code(
            "from pathlib import Path\n"
            "import json, os, shutil, subprocess, sys, zipfile\n"
            "import torch\n"
            "IS_KAGGLE = Path('/kaggle/input').exists() and Path('/kaggle/working').exists()\n"
            "IS_COLAB = Path('/var/colab/hostname').exists() and not IS_KAGGLE\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('Select a GPU accelerator before continuing.')\n"
            "gpu = torch.cuda.get_device_properties(0)\n"
            "print('Platform:', 'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'local')\n"
            "print('GPU:', gpu.name, '| VRAM GiB:', round(gpu.total_memory / 2**30, 1))\n"
        ),
        code(
            "WORK = Path('/content/bikv_v19_stage_i_b_work') if IS_COLAB else Path('/kaggle/working/bikv_v19_stage_i_b_work')\n"
            "if IS_COLAB:\n"
            "    from google.colab import drive, files\n"
            "    drive.mount('/content/drive')\n"
            "    MIRROR = Path('/content/drive/MyDrive/bikv_runs')\n"
            f"    candidates = list(Path('/content').glob('{ARCHIVE}*.zip'))\n"
            "    if not candidates:\n"
            "        files.upload()\n"
            f"        candidates = list(Path('/content').glob('{ARCHIVE}*.zip'))\n"
            "elif IS_KAGGLE:\n"
            "    MIRROR = None\n"
            f"    candidates = list(Path('/kaggle/input').rglob('{ARCHIVE}*.zip'))\n"
            f"    candidates += list(Path('/kaggle/working').glob('{ARCHIVE}*.zip'))\n"
            "else:\n"
            "    MIRROR = None\n"
            f"    candidates = list(Path.cwd().glob('{ARCHIVE}*.zip'))\n"
            "SOURCE = WORK / 'source_v19'\n"
            "if SOURCE.exists():\n"
            "    shutil.rmtree(SOURCE)\n"
            "SOURCE.mkdir(parents=True, exist_ok=True)\n"
            "if candidates:\n"
            "    archive_path = max(candidates, key=lambda path: path.stat().st_mtime)\n"
            "    with zipfile.ZipFile(archive_path) as archive:\n"
            "        archive.extractall(SOURCE)\n"
            "elif IS_KAGGLE:\n"
            "    projects = [p.parent for p in Path('/kaggle/input').rglob('pyproject.toml') if (p.parent / 'src' / 'bikv').is_dir()]\n"
            "    if len(projects) != 1:\n"
            "        raise FileNotFoundError(f'Expected one extracted BIKV project, found {projects}')\n"
            "    shutil.copytree(projects[0], SOURCE / 'boundary_interaction_kv', dirs_exist_ok=True)\n"
            "else:\n"
            f"    raise FileNotFoundError('Upload {ARCHIVE}.zip')\n"
            "projects = list(SOURCE.rglob('pyproject.toml'))\n"
            "if len(projects) != 1:\n"
            "    raise RuntimeError(f'Expected one project, found {len(projects)}')\n"
            "PROJECT = projects[0].parent\n"
            "print('Project:', PROJECT)\n"
        ),
        code(
            "os.environ['HF_HOME'] = str(WORK / 'hf_cache')\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT), 'pyarrow>=15,<24'], check=True)\n"
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
        ),
        markdown(
            "## Download or reuse the pinned official source\n\n"
            "The source is stored on Drive in Colab. Selection uses the same seed and filter as Stage I, but starts "
            "at offset 24; overlap with the checked-in Stage I manifest is a fatal error.\n"
        ),
        code(
            "DATA = (Path('/content/drive/MyDrive/bikv_datasets/v18_longbench_2wiki') if IS_COLAB else WORK / 'data')\n"
            "stream([sys.executable, str(PROJECT / 'scripts' / 'prepare_longbench_2wiki.py'), '--output', str(DATA), "
            "'--revision', '0962550175777a5b10df401ca715e01c0bb3ee8d'])\n"
            "print((DATA / 'source_manifest.json').read_text())\n"
        ),
        code(
            "base = json.loads((PROJECT / 'configs' / 'v19_stage_i_b_longbench_2wiki.json').read_text())\n"
            "base['data_path'] = str(DATA / '2wikimqa_e_source.jsonl')\n"
            "base['longbench_source_manifest'] = str(DATA / 'source_manifest.json')\n"
            "base['stage_h_frozen_models_path'] = str(PROJECT / 'configs' / 'v17_stage_h_frozen_models.json')\n"
            "base['longbench_prior_selection_manifest'] = str(PROJECT / 'configs' / 'v18_stage_i_selection_manifest.json')\n"
            "base['longbench_expected_selection_manifest'] = str(PROJECT / 'configs' / 'v19_stage_i_b_selection_manifest.json')\n"
            "base['output_root'] = str(WORK / 'runs')\n"
            "base['mirror_root'] = str(MIRROR) if MIRROR else None\n"
            "runtime_path = WORK / 'runtime_stage_i_b_longbench.json'\n"
            "runtime_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "runtime_path.write_text(json.dumps(base, indent=2), encoding='utf-8')\n"
            "print(runtime_path.read_text())\n"
        ),
        markdown(
            "## Run or resume\n\n"
            "Re-run this cell after a disconnect. Completed captures and method rows are restored from Drive and skipped. "
            "An artifact is mirrored after every two completed samples.\n"
        ),
        code(
            "stream([sys.executable, '-u', str(PROJECT / 'scripts' / 'stage_i_longbench.py'), '--config', str(runtime_path)])\n"
        ),
        code(
            "run_root = Path(base['output_root']) / base['run_id']\n"
            "stream([sys.executable, str(PROJECT / 'scripts' / 'analyze_stage_i_longbench.py'), str(run_root), "
            "'--seed', str(base['seed']), '--expected', str(base['longbench_samples']), "
            "'--budget', str(base['budget_blocks']), '--confirmatory'])\n"
            "sys.path.insert(0, str(PROJECT / 'src'))\n"
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
            "    print('Kaggle artifact is in /kaggle/working. Save the notebook version to retain it.')\n"
        ),
    ]
    notebook = {
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
    destination = root / "notebooks" / name
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
