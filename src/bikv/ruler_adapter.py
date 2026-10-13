from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RULER_QUERY_MARKER = "BIKV_RULER_QUERY_MARKER_7C91"


@dataclass
class RulerSample:
    sample_id: str
    story_id: str
    blocks: list[str]
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_blocks: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def convert_ruler_row(
    row: dict[str, Any], tokenizer: Any, block_size: int, length_label: int, source_revision: str
) -> RulerSample:
    text = str(row["input"])
    if text.count(RULER_QUERY_MARKER) != 1:
        raise ValueError("RULER input does not contain the unique BIKV query marker")
    document, question = text.split(RULER_QUERY_MARKER)
    document_ids = list(tokenizer.encode(document.strip(), add_special_tokens=False))
    if not document_ids:
        raise ValueError("RULER document tokenization is empty")
    token_blocks = [document_ids[index : index + block_size] for index in range(0, len(document_ids), block_size)]
    references = [str(value).strip() for value in row["outputs"]]
    if not references or any(not value for value in references):
        raise ValueError("RULER references must be non-empty")
    index = int(row["index"])
    return RulerSample(
        sample_id=f"ruler-vt-{length_label:05d}-{index:04d}",
        story_id=f"ruler-vt-{length_label:05d}-{index:04d}",
        blocks=[f"PRETOKENIZED_BLOCK_{block_index:04d}" for block_index in range(len(token_blocks))],
        question=question.strip(),
        answer=" ".join(references),
        answer_aliases=[" ".join(references)],
        metadata={
            "benchmark": "NVIDIA/RULER",
            "task": "variable_tracking",
            "length_label": length_label,
            "official_length": int(row.get("length", len(document_ids))),
            "official_index": index,
            "references": references,
            "block_token_ids": token_blocks,
            "source_revision": source_revision,
            "counterfactual_group": f"ruler-{length_label:05d}-{index:04d}",
            "evidence_order": "not_available",
        },
    )


def load_ruler_samples(
    path: str | Path, tokenizer: Any, block_size: int, length_label: int, source_revision: str
) -> list[RulerSample]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(convert_ruler_row(json.loads(line), tokenizer, block_size, length_label, source_revision))
    return rows


def ruler_metrics(prediction: str, references: list[str]) -> dict[str, float | bool | list[str]]:
    expected = {value.casefold() for value in references}
    predicted = {value.casefold() for value in re.findall(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])", prediction)}
    overlap = expected & predicted
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_recall": recall,
        "set_precision": precision,
        "set_f1": f1,
        "set_exact_match": predicted == expected,
        "predicted_variables": sorted(predicted),
        "reference_variables": sorted(expected),
    }
