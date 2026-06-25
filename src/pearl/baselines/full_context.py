"""Full-context baseline."""

from __future__ import annotations

from pearl.baselines.common import documents_to_text, pack_chunks
from pearl.data.schema import Chunk, LongContextExample
from pearl.preprocessing.token_counting import count_tokens


def full_context_pack(example: LongContextExample) -> tuple[str, object]:
    """Pack all documents as one chunk."""

    text = documents_to_text(example)
    chunk = Chunk(
        chunk_id="full_context",
        doc_id="__all__",
        text=text,
        token_len=count_tokens(text),
        section_title=None,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
    )
    return pack_chunks(example, [chunk], method="full_context")
