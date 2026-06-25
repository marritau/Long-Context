"""Aggregate metric rows."""

from __future__ import annotations

from collections import defaultdict


def mean_by_key(rows: list[dict], group_key: str, metric_keys: list[str]) -> list[dict]:
    """Compute mean metrics grouped by a key."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: list[dict] = []
    for group, items in grouped.items():
        result = {group_key: group}
        for metric in metric_keys:
            values = [float(item[metric]) for item in items if metric in item]
            result[metric] = sum(values) / len(values) if values else 0.0
        out.append(result)
    return out
