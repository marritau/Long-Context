"""Ordering strategies for packed evidence chunks."""

from __future__ import annotations

import random

from pearl.data.schema import PackedChunk
from pearl.packing.position_utility import lost_middle_utility, normalized_position


def position_aware_order(
    chunks: list[PackedChunk],
    importances: list[float] | None = None,
    mode: str = "pearl",
    rho: float = 0.35,
    kappa: float = 2.0,
    seed: int = 42,
) -> list[PackedChunk]:
    """Order chunks with document, relevance, random, head-tail, or PEARL modes."""

    if not chunks:
        return []
    if importances is None:
        importances = [chunk.score for chunk in chunks]
    if len(importances) != len(chunks):
        raise ValueError("importances length must match chunks length")

    indexed = list(enumerate(chunks))
    mode = mode.lower()
    if mode in {"document", "doc"}:
        ordered = [chunk for _, chunk in sorted(indexed, key=lambda item: item[1].chunk.chunk_index)]
    elif mode in {"relevance", "relevance_desc"}:
        ordered = [
            chunks[idx]
            for idx in sorted(range(len(chunks)), key=lambda idx: importances[idx], reverse=True)
        ]
    elif mode == "random":
        ordered = chunks[:]
        random.Random(seed).shuffle(ordered)
    elif mode == "head_tail":
        ordered = _head_tail_order(chunks, importances)
    elif mode == "pearl":
        ordered = _pearl_order(chunks, importances, rho, kappa)
    else:
        raise ValueError(f"unknown ordering mode: {mode}")

    for idx, chunk in enumerate(ordered):
        chunk.position = normalized_position(idx, len(ordered))
    return ordered


def _head_tail_order(chunks: list[PackedChunk], importances: list[float]) -> list[PackedChunk]:
    ranked = sorted(range(len(chunks)), key=lambda idx: importances[idx], reverse=True)
    slots: list[PackedChunk | None] = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1
    place_left = True
    for idx in ranked:
        if place_left:
            slots[left] = chunks[idx]
            left += 1
        else:
            slots[right] = chunks[idx]
            right -= 1
        place_left = not place_left
    return [chunk for chunk in slots if chunk is not None]


def _pearl_order(
    chunks: list[PackedChunk],
    importances: list[float],
    rho: float,
    kappa: float,
) -> list[PackedChunk]:
    n = len(chunks)
    chunk_order = sorted(range(n), key=lambda idx: importances[idx], reverse=True)
    slot_order = sorted(
        range(n),
        key=lambda idx: lost_middle_utility(normalized_position(idx, n), rho, kappa),
        reverse=True,
    )
    slots: list[PackedChunk | None] = [None] * n
    for chunk_idx, slot_idx in zip(chunk_order, slot_order, strict=True):
        slots[slot_idx] = chunks[chunk_idx]
    return [chunk for chunk in slots if chunk is not None]
