"""Self-Route-inspired baseline."""

from __future__ import annotations

from pearl.baselines.full_context import full_context_pack
from pearl.baselines.hybrid_rag import hybrid_rrf_pack
from pearl.data.schema import LongContextExample
from pearl.preprocessing.token_counting import count_tokens


def self_route_pack(
    example: LongContextExample,
    long_context_threshold: int = 12000,
    rag_budget: int = 8192,
) -> tuple[str, object]:
    """Choose full-context mode for shorter inputs and RAG otherwise."""

    total_tokens = sum(count_tokens(doc.text) for doc in example.documents)
    if total_tokens <= long_context_threshold:
        return full_context_pack(example)
    return hybrid_rrf_pack(example, top_k=max(1, rag_budget // 768))
