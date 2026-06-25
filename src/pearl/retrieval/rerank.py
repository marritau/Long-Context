"""Reranking interfaces and lightweight fallback reranker."""

from __future__ import annotations

from collections import Counter

from pearl.data.schema import Chunk
from pearl.preprocessing.token_counting import simple_tokenize


def lexical_overlap_score(query: str, text: str) -> float:
    """A small cross-encoder-style fallback score based on token overlap."""

    q_tokens = Counter(simple_tokenize(query))
    t_tokens = Counter(simple_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    common = sum((q_tokens & t_tokens).values())
    precision = common / sum(t_tokens.values())
    recall = common / sum(q_tokens.values())
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    phrase_bonus = 0.15 if query.lower() in text.lower() else 0.0
    return f1 + phrase_bonus


def rerank_scores(query: str, chunks: list[Chunk]) -> list[float]:
    """Return fallback reranker scores for chunks."""

    return [lexical_overlap_score(query, chunk.text) for chunk in chunks]
