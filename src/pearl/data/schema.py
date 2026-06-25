"""Canonical data structures for PEARL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Section:
    """A structured section inside a document."""

    title: str | None
    text: str
    section_path: list[str] = field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """A source document used by a long-context example."""

    doc_id: str
    title: str | None
    text: str
    sections: list[Section] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceSpan:
    """Gold or predicted evidence span."""

    doc_id: str
    text: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LongContextExample:
    """A normalized QA example with one or more long documents."""

    example_id: str
    dataset: str
    question: str
    documents: list[Document]
    answer: str | list[str]
    gold_evidence: list[EvidenceSpan] | None = None
    task_type: str = "qa"
    context_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A chunk produced from a document."""

    chunk_id: str
    doc_id: str
    text: str
    token_len: int
    section_title: str | None
    chunk_index: int
    char_start: int
    char_end: int
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PackedChunk:
    """A chunk after optional compression and prompt-position assignment."""

    chunk: Chunk
    text: str
    token_len: int
    original_token_len: int
    score: float = 0.0
    position: float | None = None
    evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PackedContext:
    """The assembled evidence package that will be placed into a prompt."""

    example_id: str
    method: str
    chunks: list[PackedChunk]
    total_tokens: int
    compression_ratio: float
    ordering_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def evidence_text(self) -> str:
        """Return prompt-ready numbered evidence text."""

        lines: list[str] = []
        for idx, packed in enumerate(self.chunks, start=1):
            lines.append(f"[{idx}] {packed.text}")
        return "\n\n".join(lines)
