"""Hybrid RRF baseline."""

from __future__ import annotations

from pearl.baselines.common import pack_chunks
from pearl.data.schema import LongContextExample
from pearl.preprocessing.chunking import chunk_documents
from pearl.retrieval.bm25 import BM25Index
from pearl.retrieval.dense import HashingDenseRetriever
from pearl.retrieval.hybrid import hybrid_rrf_ids


def hybrid_rrf_pack(
    example: LongContextExample,
    top_k: int = 12,
    chunk_size: int = 768,
    overlap: int = 128,
    rrf_k: int = 60,
) -> tuple[str, object]:
    """Pack hybrid BM25+dense RRF top-k chunks."""

    chunks = chunk_documents(example.documents, chunk_size=chunk_size, overlap=overlap)
    bm25_scores = BM25Index(chunks).score(example.question)
    dense_scores = HashingDenseRetriever(chunks).score(example.question)
    ids = hybrid_rrf_ids(bm25_scores, dense_scores, top_m=top_k, rrf_k=rrf_k)
    return pack_chunks(example, [chunks[idx] for idx in ids], method="hybrid_rrf")
