from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StorySample:
    sample_id: str
    story_id: str
    split: str
    variant: str
    blocks: list[str]
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_blocks: list[int]
    manipulated_boundary: int
    boundary_family: str = "false_split"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.variant not in {"neutral", "targeted", "sham"}:
            raise ValueError(f"Unsupported variant: {self.variant}")
        if not self.blocks:
            raise ValueError("A sample must contain at least one block")
        if len(self.supporting_blocks) < 2:
            raise ValueError("At least two supporting blocks are required")
        if len(set(self.supporting_blocks)) != len(self.supporting_blocks):
            raise ValueError("Supporting block indices must be unique")
        if any(index < 0 or index >= len(self.blocks) for index in self.supporting_blocks):
            raise ValueError("Supporting block index is outside the document")
        if not 0 <= self.manipulated_boundary < len(self.blocks) - 1:
            raise ValueError("Boundary index must identify an adjacent block pair")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StorySample":
        sample = cls(**value)
        sample.validate()
        return sample


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(output)


def load_samples(path: str | Path) -> list[StorySample]:
    samples: list[StorySample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(StorySample.from_dict(json.loads(line)))
    return samples
