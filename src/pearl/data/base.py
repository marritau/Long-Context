"""Base dataset loading helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pearl.data.schema import Document, EvidenceSpan, LongContextExample
from pearl.utils.io import read_jsonl


class DatasetLoader(ABC):
    """Common interface for dataset loaders."""

    dataset_name: str

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    def load(self, limit: int | None = None) -> list[LongContextExample]:
        """Load examples in the canonical schema."""


class JsonlDatasetLoader(DatasetLoader):
    """Load already-normalized examples from JSONL.

    This loader is useful for the MVP and for processed dataset files. Each row
    may contain either a list of `documents` objects or a simple `context` field.
    """

    dataset_name = "jsonl"

    def load(self, limit: int | None = None) -> list[LongContextExample]:
        rows = read_jsonl(self.path)
        if limit is not None:
            rows = rows[:limit]
        return [self._row_to_example(row, idx) for idx, row in enumerate(rows)]

    def _row_to_example(self, row: dict, idx: int) -> LongContextExample:
        docs = row.get("documents")
        if docs is None:
            context = row.get("context", "")
            docs = [{"doc_id": row.get("doc_id", "doc_0"), "title": None, "text": context}]

        documents = [
            Document(
                doc_id=str(doc.get("doc_id", f"doc_{doc_idx}")),
                title=doc.get("title"),
                text=doc.get("text", ""),
                sections=None,
                metadata=doc.get("metadata", {}),
            )
            for doc_idx, doc in enumerate(docs)
        ]

        gold_rows = row.get("gold_evidence") or row.get("evidence") or []
        gold = [
            EvidenceSpan(
                doc_id=str(item.get("doc_id", "doc_0")),
                text=item.get("text"),
                char_start=item.get("char_start"),
                char_end=item.get("char_end"),
                section_title=item.get("section_title"),
                metadata=item.get("metadata", {}),
            )
            for item in gold_rows
        ]

        return LongContextExample(
            example_id=str(row.get("example_id", row.get("id", idx))),
            dataset=str(row.get("dataset", self.dataset_name)),
            question=row.get("question", ""),
            documents=documents,
            answer=row.get("answer", ""),
            gold_evidence=gold or None,
            task_type=row.get("task_type", "qa"),
            context_length=row.get("context_length"),
            metadata=row.get("metadata", {}),
        )
