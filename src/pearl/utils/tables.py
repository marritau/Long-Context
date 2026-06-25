"""Table formatting helpers."""

from __future__ import annotations


def rows_to_markdown(rows: list[dict], columns: list[str]) -> str:
    """Render rows as a simple Markdown table."""

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])
