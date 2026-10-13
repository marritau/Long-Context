from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any


METHODS = (
    "full",
    "relation_graph",
    "graph_matched_unary_ridge_v1",
    "graph_arbitrary_pair_ridge_v1",
    "graph_pair_shuffled_v1",
    "snapkv_upstream_query_aware",
)


def load_rows(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for method in METHODS:
        directory = root / "stage_i" / "longbench_2wiki" / method
        rows = {}
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                row = json.loads(path.read_text(encoding="utf-8"))
                rows[row["sample_id"]] = row
        result[method] = rows
    return result


def bootstrap(left, right, seed: int, repeats: int = 10000) -> dict[str, Any]:
    sample_ids = sorted(set(left) & set(right))
    deltas = [
        float(left[sample]["qa_metrics"]["answer_f1"])
        - float(right[sample]["qa_metrics"]["answer_f1"])
        for sample in sample_ids
    ]
    if not deltas:
        return {"difference": None, "ci95": None, "samples": 0}
    rng = random.Random(seed)
    draws = sorted(mean(rng.choice(deltas) for _ in deltas) for _ in range(repeats))
    return {
        "difference": mean(deltas),
        "ci95": [draws[int(0.025 * repeats)], draws[int(0.975 * repeats) - 1]],
        "samples": len(deltas),
        "pairwise_wins": sum(value > 0 for value in deltas),
        "pairwise_losses": sum(value < 0 for value in deltas),
        "pairwise_ties": sum(value == 0 for value in deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage I LongBench 2Wiki")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--expected", type=int, default=24)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows = load_rows(root)
    for method, values in rows.items():
        if len(values) != args.expected:
            raise ValueError(f"{method} has {len(values)} rows, expected {args.expected}")
        for row in values.values():
            selected = row.get("selected_blocks")
            if selected is not None and (len(selected) != args.budget or len(set(selected)) != args.budget):
                raise ValueError(f"Invalid mask in {method} {row['sample_id']}")

    summary = {}
    lines = [
        "# Stage I LongBench-E 2Wiki", "", "Integrity: PASS", "",
        "| Method | N | Answer F1 | Exact match |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = list(rows[method].values())
        item = {
            "n": len(values),
            "answer_f1": mean(float(row["qa_metrics"]["answer_f1"]) for row in values),
            "answer_exact_match": mean(float(row["qa_metrics"]["answer_exact_match"]) for row in values),
        }
        summary[method] = item
        lines.append(
            f"| {method} | {item['n']} | {item['answer_f1']:.1%} | {item['answer_exact_match']:.1%} |"
        )
    comparisons = {
        "pair_minus_matched_unary": bootstrap(
            rows["graph_arbitrary_pair_ridge_v1"], rows["graph_matched_unary_ridge_v1"], args.seed
        ),
        "pair_minus_shuffled": bootstrap(
            rows["graph_arbitrary_pair_ridge_v1"], rows["graph_pair_shuffled_v1"], args.seed + 1
        ),
        "pair_minus_snapkv": bootstrap(
            rows["graph_arbitrary_pair_ridge_v1"], rows["snapkv_upstream_query_aware"], args.seed + 2
        ),
        "relation_graph_minus_matched_unary": bootstrap(
            rows["relation_graph"], rows["graph_matched_unary_ridge_v1"], args.seed + 3
        ),
        "snapkv_minus_full": bootstrap(
            rows["snapkv_upstream_query_aware"], rows["full"], args.seed + 4
        ),
    }
    lines.extend(["", "## Paired F1 comparisons", "", "```json", json.dumps(comparisons, indent=2), "```"])
    report = {
        "stage": "I",
        "benchmark": "LongBench-E",
        "task": "2wikimqa_e",
        "integrity": "PASS",
        "exploratory": not args.confirmatory,
        "confirmatory_replication": args.confirmatory,
        "summary": summary,
        "comparisons": comparisons,
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
