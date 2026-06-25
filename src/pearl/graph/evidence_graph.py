"""Evidence graph construction."""

from __future__ import annotations

from pearl.data.schema import Chunk
from pearl.retrieval.dense import cosine


def build_evidence_graph(
    chunks: list[Chunk],
    vectors: list[list[float]] | None = None,
    semantic_weight: float = 0.60,
    adjacency_weight: float = 0.25,
    section_weight: float = 0.15,
    semantic_threshold: float = 0.05,
) -> list[list[float]]:
    """Build a row-normalized chunk graph.

    Edge types:
    - semantic similarity;
    - adjacency inside the same document;
    - same-section links.
    """

    n = len(chunks)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, left in enumerate(chunks):
        for j, right in enumerate(chunks):
            if i == j:
                continue
            weight = 0.0
            if vectors is not None:
                sim = max(0.0, cosine(vectors[i], vectors[j]))
                if sim >= semantic_threshold:
                    weight += semantic_weight * sim
            if left.doc_id == right.doc_id and abs(left.chunk_index - right.chunk_index) == 1:
                weight += adjacency_weight
            if (
                left.doc_id == right.doc_id
                and left.section_title
                and left.section_title == right.section_title
            ):
                weight += section_weight
            matrix[i][j] = weight
    return row_normalize(matrix)


def row_normalize(matrix: list[list[float]]) -> list[list[float]]:
    """Normalize each row to sum to 1, falling back to self-loops."""

    normalized: list[list[float]] = []
    n = len(matrix)
    for idx, row in enumerate(matrix):
        total = sum(row)
        if total <= 0.0:
            fallback = [0.0] * n
            if n:
                fallback[idx] = 1.0
            normalized.append(fallback)
        else:
            normalized.append([value / total for value in row])
    return normalized
