"""Small BM25 implementation used as a dependency-light baseline."""

from __future__ import annotations

import math
from collections import Counter

from pearl.data.schema import Chunk
from pearl.preprocessing.token_counting import simple_tokenize


class BM25Index:
    """BM25 index over PEARL chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [simple_tokenize(chunk.text) for chunk in chunks]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self.doc_tokens)
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        return {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> list[float]:
        """Return BM25 scores for all indexed chunks."""

        query_terms = simple_tokenize(query)
        scores: list[float] = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_lens, strict=True):
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
            scores.append(score)
        return scores

    def top_k(self, query: str, k: int) -> list[int]:
        """Return top-k chunk indices for a query."""

        scores = self.score(query)
        return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
