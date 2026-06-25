"""Dataset evaluation scaffold."""

from __future__ import annotations

import argparse

from pearl.data.base import JsonlDatasetLoader
from pearl.packing.pack_builder import pearl_pack
from pearl.utils.io import read_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Normalized PEARL JSONL file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/metrics/dataset_eval.jsonl")
    args = parser.parse_args()

    config = read_yaml(args.config)
    examples = JsonlDatasetLoader(args.data).load(limit=args.limit)
    rows = []
    for example in examples:
        _, packed = pearl_pack(example, config)
        rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "method": packed.method,
                "tokens": packed.total_tokens,
                "num_chunks": len(packed.chunks),
            }
        )
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
