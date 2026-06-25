"""Build chunk indexes for normalized PEARL JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

from pearl.data.base import JsonlDatasetLoader
from pearl.preprocessing.chunking import chunk_documents
from pearl.retrieval.index_store import save_chunks
from pearl.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Normalized JSONL examples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="data/indexes")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    chunk_cfg = config.get("chunking", {})
    examples = JsonlDatasetLoader(args.data).load(limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for example in examples:
        chunks = chunk_documents(
            example.documents,
            chunk_size=chunk_cfg.get("chunk_size", 768),
            overlap=chunk_cfg.get("chunk_overlap", 128),
            structure_aware=chunk_cfg.get("structure_aware", True),
        )
        save_chunks(out_dir / f"{example.example_id}.chunks.json", chunks)


if __name__ == "__main__":
    main()
