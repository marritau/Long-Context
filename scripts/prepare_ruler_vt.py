from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bikv.ruler_adapter import RULER_QUERY_MARKER


ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text above, "
    "{num_v} variables are assigned the value {query}, they are: "
)
TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    + RULER_QUERY_MARKER
    + "\nQuestion: Find all variables that are assigned the value {query} in the text above."
    + ANSWER_PREFIX
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(repo: Path, output: Path, model: str, length: int, samples: int, seed: int) -> list[str]:
    return [
        sys.executable,
        str(repo / "scripts" / "data" / "synthetic" / "variable_tracking.py"),
        "--save_dir", str(output),
        "--save_name", f"vt_{length}",
        "--tokenizer_path", model,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(length),
        "--tokens_to_generate", "64",
        "--num_samples", str(samples),
        "--random_seed", str(seed),
        "--num_chains", "1",
        "--num_hops", "4",
        "--template", TEMPLATE,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pinned official RULER Variable Tracking data")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="c3f5e3b")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if not head.startswith(args.revision):
        raise ValueError(f"RULER revision mismatch: expected {args.revision}, found {head}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "NVIDIA/RULER", "revision": head, "datasets": {}}
    for length in args.lengths:
        destination = args.output / f"vt_{length}" / "validation.jsonl"
        if not destination.exists():
            subprocess.run(
                command_for(args.repo, args.output, args.model, length, args.samples, args.seed + length),
                check=True,
            )
        rows = sum(1 for line in destination.open("r", encoding="utf-8") if line.strip())
        if rows != args.samples:
            raise ValueError(f"RULER {length} contains {rows} rows, expected {args.samples}")
        manifest["datasets"][str(length)] = {
            "path": str(destination), "rows": rows, "sha256": sha256(destination)
        }
    (args.output / "ruler_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
