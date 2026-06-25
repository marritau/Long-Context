"""Create aggregate CSV-like Markdown tables from metric JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from pearl.evaluation.aggregate import mean_by_key
from pearl.utils.io import read_jsonl
from pearl.utils.tables import rows_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--group_key", default="method")
    parser.add_argument("--metric_keys", nargs="+", default=["tokens"])
    parser.add_argument("--out", default="results/tables/aggregate.md")
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    table_rows = mean_by_key(rows, args.group_key, args.metric_keys)
    columns = [args.group_key, *args.metric_keys]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rows_to_markdown(table_rows, columns), encoding="utf-8")


if __name__ == "__main__":
    main()
