from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bikv.g1b_data import generate_g1b_dataset
from bikv.schema import write_jsonl


AUDIT_PREFIXES = ("Audit note.", "Maintenance log.", "Archive cross-check.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independent stratified Stage G1b datasets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stories", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--blocks", type=int, nargs="+", default=[21, 42, 84])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"seed": args.seed, "stories_per_length": args.stories, "lengths": {}}
    for n_blocks in args.blocks:
        samples = generate_g1b_dataset(args.stories, n_blocks, args.seed + n_blocks * 100003)
        path = args.output / f"g1b_{n_blocks:03d}_blocks.jsonl"
        write_jsonl(path, (sample.to_dict() for sample in samples))
        positions = Counter(index for sample in samples for index in sample.supporting_blocks)
        conditions = Counter(str(sample.metadata["evidence_condition"]) for sample in samples)
        families = {str(sample.metadata["counterfactual_group"]) for sample in samples}
        leaked_audits = sum(
            str(sample.metadata["target_lab"]) in block
            for sample in samples
            for block in sample.blocks
            if block.startswith(AUDIT_PREFIXES)
        )
        if leaked_audits != 0:
            raise ValueError(f"Target-laboratory audit leakage detected for n_blocks={n_blocks}")
        audit["lengths"][str(n_blocks)] = {
            "path": str(path),
            "sha256": sha256(path),
            "stories": len(samples),
            "families": len(families),
            "conditions": dict(conditions),
            "min_support_position": min(positions),
            "max_support_position": max(positions),
            "covered_support_positions": len(positions),
            "audit_target_lab_mentions": leaked_audits,
        }
    (args.output / "g1b_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
