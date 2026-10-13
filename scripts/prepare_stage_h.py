from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bikv.chain_data import audit_chain_dataset, generate_chain_dataset
from bikv.schema import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Stage H controlled chain dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stories", type=int, default=36)
    parser.add_argument("--blocks", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    samples = generate_chain_dataset(args.stories, args.blocks, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "stage_h_chains_4k.jsonl"
    write_jsonl(path, (sample.to_dict() for sample in samples))
    audit = audit_chain_dataset(samples)
    audit.update(
        {
            "seed": args.seed,
            "n_blocks": args.blocks,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": path.name,
        }
    )
    (args.output / "stage_h_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
