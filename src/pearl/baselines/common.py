"""Shared baseline helpers."""

from __future__ import annotations

from pearl.data.schema import Chunk, LongContextExample, PackedChunk, PackedContext
from pearl.packing.ordering import position_aware_order
from pearl.packing.prompt_templates import build_qa_prompt
from pearl.preprocessing.token_counting import count_tokens


def documents_to_text(example: LongContextExample) -> str:
    """Concatenate example documents."""

    parts = []
    for doc in example.documents:
        title = f"# {doc.title}\n" if doc.title else ""
        parts.append(f"{title}{doc.text}")
    return "\n\n".join(parts)


def pack_chunks(
    example: LongContextExample,
    chunks: list[Chunk],
    method: str,
    ordering: str = "document",
) -> tuple[str, PackedContext]:
    """Convert chunks into prompt and PackedContext."""

    packed = [
        PackedChunk(
            chunk=chunk,
            text=chunk.text,
            token_len=chunk.token_len,
            original_token_len=chunk.token_len,
        )
        for chunk in chunks
    ]
    ordered = position_aware_order(packed, mode=ordering)
    context = PackedContext(
        example_id=example.example_id,
        method=method,
        chunks=ordered,
        total_tokens=sum(item.token_len for item in ordered),
        compression_ratio=1.0 if ordered else 0.0,
        ordering_strategy=ordering,
        metadata={"prompt_tokens": count_tokens(build_qa_prompt(example.question, ordered))},
    )
    return build_qa_prompt(example.question, ordered), context
