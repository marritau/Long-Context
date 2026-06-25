"""Oracle evidence baseline."""

from __future__ import annotations

from pearl.baselines.common import pack_chunks
from pearl.data.schema import Chunk, LongContextExample
from pearl.preprocessing.token_counting import count_tokens


def oracle_pack(example: LongContextExample) -> tuple[str, object]:
    """Pack gold evidence spans when available."""

    chunks: list[Chunk] = []
    for idx, span in enumerate(example.gold_evidence or []):
        text = span.text or ""
        chunks.append(
            Chunk(
                chunk_id=f"oracle:{idx}",
                doc_id=span.doc_id,
                text=text,
                token_len=count_tokens(text),
                section_title=span.section_title,
                chunk_index=idx,
                char_start=span.char_start or 0,
                char_end=span.char_end or len(text),
            )
        )
    return pack_chunks(example, chunks, method="oracle")
