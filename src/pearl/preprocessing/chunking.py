"""Document chunking with overlap and basic structure awareness."""

from __future__ import annotations

from pearl.data.schema import Chunk, Document, Section
from pearl.preprocessing.text_cleaning import clean_text
from pearl.preprocessing.token_counting import count_tokens, iter_token_spans


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 768,
    overlap: int = 128,
    structure_aware: bool = True,
) -> list[Chunk]:
    """Chunk documents into token windows with overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[Chunk] = []
    for doc in documents:
        sections = doc.sections if structure_aware and doc.sections else None
        if sections:
            for section in sections:
                chunks.extend(_chunk_section(doc, section, chunk_size, overlap, len(chunks)))
        else:
            section = Section(
                title=doc.title,
                text=doc.text,
                section_path=[doc.title] if doc.title else [],
                char_start=0,
                char_end=len(doc.text),
            )
            chunks.extend(_chunk_section(doc, section, chunk_size, overlap, len(chunks)))
    return chunks


def _chunk_section(
    doc: Document,
    section: Section,
    chunk_size: int,
    overlap: int,
    global_offset: int,
) -> list[Chunk]:
    section_text = clean_text(section.text)
    token_spans = list(iter_token_spans(section_text))
    if not token_spans:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    local_index = 0
    section_base = section.char_start or 0

    for start_token in range(0, len(token_spans), step):
        end_token = min(start_token + chunk_size, len(token_spans))
        selected = token_spans[start_token:end_token]
        if not selected:
            continue

        start_char = selected[0][1]
        end_char = selected[-1][2]
        text = section_text[start_char:end_char]
        absolute_start = section_base + start_char
        absolute_end = section_base + end_char
        chunk_index = global_offset + len(chunks)

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{chunk_index}",
                doc_id=doc.doc_id,
                text=text,
                token_len=count_tokens(text),
                section_title=section.title,
                section_path=section.section_path.copy(),
                chunk_index=chunk_index,
                char_start=absolute_start,
                char_end=absolute_end,
                metadata={
                    "local_chunk_index": local_index,
                    "source_title": doc.title,
                },
            )
        )
        local_index += 1
        if end_token == len(token_spans):
            break

    return chunks
