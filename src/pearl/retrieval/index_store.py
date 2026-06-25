"""Simple JSON index persistence."""

from __future__ import annotations

from pathlib import Path

from pearl.data.schema import Chunk
from pearl.utils.io import read_json, write_json


def save_chunks(path: str | Path, chunks: list[Chunk]) -> None:
    """Save chunks to JSON."""

    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "token_len": chunk.token_len,
                "section_title": chunk.section_title,
                "section_path": chunk.section_path,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "metadata": chunk.metadata,
            }
        )
    write_json(path, rows)


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunks from JSON."""

    rows = read_json(path)
    return [
        Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            token_len=row["token_len"],
            section_title=row.get("section_title"),
            section_path=row.get("section_path", []),
            chunk_index=row["chunk_index"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]
