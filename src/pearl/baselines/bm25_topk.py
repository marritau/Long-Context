"""BM25 top-k baseline."""

from __future__ import annotations

from pearl.baselines.common import pack_chunks
from pearl.data.schema import LongContextExample
from pearl.preprocessing.chunking import chunk_documents
from pearl.retrieval.bm25 import BM25Index


def bm25_topk_pack(
    example: LongContextExample,
    top_k: int = 12,
    chunk_size: int = 768,
    overlap: int = 128,
) -> tuple[str, object]:
    """Pack BM25 top-k chunks."""

    chunks = chunk_documents(example.documents, chunk_size=chunk_size, overlap=overlap)
    index = BM25Index(chunks)
    ids = index.top_k(example.question, top_k)
    return pack_chunks(example, [chunks[idx] for idx in ids], method="bm25_topk")
