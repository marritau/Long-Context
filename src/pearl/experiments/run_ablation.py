"""Run local PEARL ablation variants on the smoke example."""

from __future__ import annotations

import argparse

from pearl.experiments.run_experiment import build_smoke_example
from pearl.packing.pack_builder import pearl_pack
from pearl.utils.io import read_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/metrics/smoke_ablation.jsonl")
    args = parser.parse_args()

    base = read_yaml(args.config)
    variants = {
        "pearl_r": {"graph": {"enabled": False}, "compression": {"enabled": False}, "ordering": {"enabled": False}},
        "pearl_rg": {"compression": {"enabled": False}, "ordering": {"enabled": False}},
        "pearl_rgc": {"ordering": {"enabled": False}},
        "pearl_rgo": {"compression": {"enabled": False}},
        "pearl_full": {},
    }
    example = build_smoke_example()
    rows = []
    for name, override in variants.items():
        cfg = _deep_merge(base, override)
        _, packed = pearl_pack(example, cfg, method=name)
        rows.append(
            {
                "method": name,
                "num_chunks": len(packed.chunks),
                "tokens": packed.total_tokens,
                "compression_ratio": packed.compression_ratio,
                "ordering": packed.ordering_strategy,
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
