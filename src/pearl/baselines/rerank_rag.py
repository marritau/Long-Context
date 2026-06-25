"""Hybrid retrieval plus fallback reranking baseline."""

from __future__ import annotations

from pearl.baselines.common import pack_chunks
from pearl.data.schema import LongContextExample
from pearl.preprocessing.chunking import chunk_documents
from pearl.retrieval.bm25 import BM25Index
from pearl.retrieval.dense import HashingDenseRetriever
from pearl.retrieval.hybrid import hybrid_rrf_ids
from pearl.retrieval.rerank import rerank_scores


def hybrid_rerank_pack(
    example: LongContextExample,
    top_m: int = 80,
    top_k: int = 12,
    chunk_size: int = 768,
    overlap: int = 128,
) -> tuple[str, object]:
    """Pack top-k chunks after hybrid retrieval and reranking."""

    chunks = chunk_documents(example.documents, chunk_size=chunk_size, overlap=overlap)
    bm25_scores = BM25Index(chunks).score(example.question)
    dense_scores = HashingDenseRetriever(chunks).score(example.question)
    candidates = hybrid_rrf_ids(bm25_scores, dense_scores, top_m=min(top_m, len(chunks)))
    scores = rerank_scores(example.question, [chunks[idx] for idx in candidates])
    ranked_candidates = [
        candidates[idx] for idx in sorted(range(len(candidates)), key=lambda idx: scores[idx], reverse=True)
    ]
    return pack_chunks(example, [chunks[idx] for idx in ranked_candidates[:top_k]], method="hybrid_rerank")
