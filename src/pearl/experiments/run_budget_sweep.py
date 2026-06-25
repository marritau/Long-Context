"""Run budget sweep on the smoke example."""

from __future__ import annotations

import argparse

from pearl.experiments.run_experiment import build_smoke_example
from pearl.packing.pack_builder import pearl_pack
from pearl.utils.io import read_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/metrics/smoke_budget_sweep.jsonl")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    args = parser.parse_args()

    base = read_yaml(args.config)
    example = build_smoke_example()
    rows = []
    for budget in args.budgets:
        cfg = _deep_merge(base, {"compression": {"final_budget": budget}})
        _, packed = pearl_pack(example, cfg)
        rows.append(
            {
                "budget": budget,
                "tokens": packed.total_tokens,
                "compression_ratio": packed.compression_ratio,
                "num_chunks": len(packed.chunks),
            }
        )
    write_jsonl(args.out, rows)


def _deep_merge(left: dict, right: dict) -> dict:
    out = {**left}
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
