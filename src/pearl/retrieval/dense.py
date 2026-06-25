"""Deterministic dense retrieval fallback.

This module deliberately avoids downloads. It approximates semantic retrieval
with signed hashing vectors so the rest of the PEARL pipeline can be tested.
For full experiments, replace this class with SentenceTransformers/BGE.
"""

from __future__ import annotations

import hashlib
import math

from pearl.data.schema import Chunk
from pearl.preprocessing.token_counting import simple_tokenize


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for dense vectors."""

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HashingDenseRetriever:
    """A deterministic hashing-vector retriever."""

    def __init__(self, chunks: list[Chunk], dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.chunks = chunks
        self.dim = dim
        self.chunk_vectors = [self.embed(chunk.text) for chunk in chunks]

    def embed(self, text: str) -> list[float]:
        """Embed text into a normalized signed hashing vector."""

        vec = [0.0] * self.dim
        for token in simple_tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]

    def score(self, query: str) -> list[float]:
        """Return cosine scores between query and chunks."""

        query_vec = self.embed(query)
        return [cosine(query_vec, chunk_vec) for chunk_vec in self.chunk_vectors]

    def top_k(self, query: str, k: int) -> list[int]:
        """Return top-k chunk indices."""

        scores = self.score(query)
        return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
