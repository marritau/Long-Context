from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pinned LongBench-E 2Wiki source file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="0962550175777a5b10df401ca715e01c0bb3ee8d")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    import pyarrow.parquet as parquet

    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "2wikimqa_e_source.jsonl"
    if not source.exists():
        snapshot = Path(
            snapshot_download(
                repo_id="zai-org/LongBench",
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=["2wikimqa_e/*.parquet"],
            )
        )
        parquet_files = sorted((snapshot / "2wikimqa_e").glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("Pinned LongBench snapshot contains no 2wikimqa_e parquet files")
        rows = []
        for path in parquet_files:
            rows.extend(parquet.read_table(path).to_pylist())
        with source.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    row_count = sum(1 for line in source.open("r", encoding="utf-8") if line.strip())
    manifest = {
        "repository": "zai-org/LongBench",
        "config": "2wikimqa_e",
        "revision": args.revision,
        "rows": row_count,
        "path": str(source),
        "sha256": sha256(source),
    }
    destination = args.output / "source_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
