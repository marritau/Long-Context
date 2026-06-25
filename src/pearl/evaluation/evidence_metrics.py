"""Evidence selection metrics."""

from __future__ import annotations

from pearl.data.schema import Chunk, EvidenceSpan, PackedChunk


def evidence_recall(
    selected: list[Chunk] | list[PackedChunk],
    gold: list[EvidenceSpan] | None,
    threshold: float = 0.3,
) -> float:
    """Fraction of gold evidence spans covered by selected chunks."""

    if not gold:
        return 0.0
    hits = 0
    for span in gold:
        if any(_covers(item, span, threshold) for item in selected):
            hits += 1
    return hits / len(gold)


def evidence_precision(
    selected: list[Chunk] | list[PackedChunk],
    gold: list[EvidenceSpan] | None,
    threshold: float = 0.3,
) -> float:
    """Fraction of selected chunks overlapping any gold evidence."""

    if not selected or not gold:
        return 0.0
    hits = sum(1 for item in selected if any(_covers(item, span, threshold) for span in gold))
    return hits / len(selected)


def evidence_f1(
    selected: list[Chunk] | list[PackedChunk],
    gold: list[EvidenceSpan] | None,
    threshold: float = 0.3,
) -> float:
    """Evidence F1 from precision and recall."""

    precision = evidence_precision(selected, gold, threshold)
    recall = evidence_recall(selected, gold, threshold)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _covers(item: Chunk | PackedChunk, span: EvidenceSpan, threshold: float) -> bool:
    chunk = item.chunk if isinstance(item, PackedChunk) else item
    if chunk.doc_id != span.doc_id:
        return False
    if span.char_start is not None and span.char_end is not None:
        overlap = _interval_overlap(chunk.char_start, chunk.char_end, span.char_start, span.char_end)
        gold_len = max(1, span.char_end - span.char_start)
        return overlap / gold_len >= threshold
    if span.text:
        haystack = item.text if isinstance(item, PackedChunk) else chunk.text
        return span.text.lower() in haystack.lower()
    return False


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))
