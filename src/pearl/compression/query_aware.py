"""Query-aware sentence-level compression."""

from __future__ import annotations

from collections import Counter

from pearl.compression.budget_allocator import allocate_budgets
from pearl.compression.sentence_splitter import split_sentences
from pearl.data.schema import Chunk, PackedChunk
from pearl.preprocessing.token_counting import count_tokens, simple_tokenize, truncate_tokens


def query_aware_compress(
    question: str,
    chunks: list[Chunk],
    total_budget: int,
    importances: list[float] | None = None,
    min_chunk_budget: int = 128,
    keep_neighbors: bool = True,
    temperature: float = 1.0,
) -> list[PackedChunk]:
    """Compress chunks by keeping query-relevant sentences."""

    if importances is None:
        importances = [1.0] * len(chunks)
    budgets = allocate_budgets(total_budget, importances, min_chunk_budget, temperature)
    packed: list[PackedChunk] = []
    for chunk, budget, importance in zip(chunks, budgets, importances, strict=True):
        text, kept = compress_chunk(question, chunk.text, budget, keep_neighbors=keep_neighbors)
        packed.append(
            PackedChunk(
                chunk=chunk,
                text=text,
                token_len=count_tokens(text),
                original_token_len=chunk.token_len,
                score=importance,
                metadata={"kept_sentence_indices": kept, "budget": budget},
            )
        )
    return packed


def compress_chunk(
    question: str,
    text: str,
    budget: int,
    keep_neighbors: bool = True,
) -> tuple[str, list[int]]:
    """Compress a single chunk to a token budget."""

    if budget <= 0:
        return "", []
    if count_tokens(text) <= budget:
        return text, list(range(len(split_sentences(text))))

    sentences = split_sentences(text)
    if not sentences:
        return "", []

    raw_scores = [_sentence_score(question, sentence) for sentence in sentences]
    scores = raw_scores[:]
    for idx, score in enumerate(raw_scores):
        neighbor = 0.0
        if idx > 0:
            neighbor = max(neighbor, raw_scores[idx - 1])
        if idx + 1 < len(raw_scores):
            neighbor = max(neighbor, raw_scores[idx + 1])
        scores[idx] = score + 0.15 * neighbor

    order = sorted(range(len(sentences)), key=lambda idx: scores[idx], reverse=True)
    keep: set[int] = set()
    current_tokens = 0
    for idx in order:
        candidates = [idx]
        if keep_neighbors:
            if idx > 0:
                candidates.append(idx - 1)
            if idx + 1 < len(sentences):
                candidates.append(idx + 1)
        for cand in candidates:
            if cand in keep:
                continue
            sent_tokens = count_tokens(sentences[cand])
            if current_tokens + sent_tokens > budget and keep:
                continue
            keep.add(cand)
            current_tokens += sent_tokens
            if current_tokens >= budget:
                break
        if current_tokens >= budget:
            break

    kept_indices = sorted(keep)
    compressed = " ".join(sentences[idx] for idx in kept_indices)
    if count_tokens(compressed) > budget:
        compressed = truncate_tokens(compressed, budget)
    return compressed, kept_indices


def _sentence_score(question: str, sentence: str) -> float:
    q_tokens = Counter(simple_tokenize(question))
    s_tokens = Counter(simple_tokenize(sentence))
    if not q_tokens or not s_tokens:
        return 0.0
    common = sum((q_tokens & s_tokens).values())
    recall = common / sum(q_tokens.values())
    precision = common / sum(s_tokens.values())
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    entity_bonus = _entity_overlap(question, sentence)
    return 0.75 * f1 + 0.25 * entity_bonus


def _entity_overlap(question: str, sentence: str) -> float:
    q_entities = {tok for tok in question.split() if tok[:1].isupper()}
    if not q_entities:
        return 0.0
    s_entities = {tok for tok in sentence.split() if tok[:1].isupper()}
    return len(q_entities & s_entities) / len(q_entities)
