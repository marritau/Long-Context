"""Truncation baselines."""

from __future__ import annotations

from pearl.baselines.common import documents_to_text, pack_chunks
from pearl.data.schema import Chunk, LongContextExample
from pearl.preprocessing.token_counting import count_tokens, truncate_tokens


def first_b_pack(example: LongContextExample, budget: int) -> tuple[str, object]:
    """Pack first B tokens."""

    return _trunc_pack(example, truncate_tokens(documents_to_text(example), budget), "first_b")


def last_b_pack(example: LongContextExample, budget: int) -> tuple[str, object]:
    """Pack last B tokens."""

    return _trunc_pack(example, truncate_tokens(documents_to_text(example), budget, from_end=True), "last_b")


def first_last_pack(example: LongContextExample, budget: int) -> tuple[str, object]:
    """Pack first and last halves of the budget."""

    text = documents_to_text(example)
    half = max(1, budget // 2)
    packed = truncate_tokens(text, half) + "\n...\n" + truncate_tokens(text, budget - half, from_end=True)
    return _trunc_pack(example, packed, "first_last")


def _trunc_pack(example: LongContextExample, text: str, method: str) -> tuple[str, object]:
    chunk = Chunk(
        chunk_id=method,
        doc_id="__all__",
        text=text,
        token_len=count_tokens(text),
        section_title=None,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
    )
    return pack_chunks(example, [chunk], method=method)
