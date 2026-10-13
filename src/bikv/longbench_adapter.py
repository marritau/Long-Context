from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


@dataclass
class LongBenchSample:
    sample_id: str
    story_id: str
    blocks: list[str]
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_blocks: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_qa_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def qa_metrics(prediction: str, references: list[str]) -> dict[str, float | bool]:
    normalized_prediction = normalize_qa_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    best_f1 = 0.0
    exact = False
    for reference in references:
        normalized_reference = normalize_qa_answer(reference)
        exact = exact or normalized_prediction == normalized_reference
        reference_tokens = normalized_reference.split()
        common: dict[str, int] = {}
        for token in set(prediction_tokens) | set(reference_tokens):
            common[token] = min(prediction_tokens.count(token), reference_tokens.count(token))
        overlap = sum(common.values())
        if overlap:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(reference_tokens)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return {"answer_f1": best_f1, "answer_exact_match": exact}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_longbench_samples(
    path: str | Path,
    tokenizer: Any,
    *,
    block_size: int,
    count: int,
    seed: int,
    offset: int = 0,
    min_official_length: int = 4000,
    max_official_length: int = 8000,
    min_document_tokens: int = 3072,
    max_document_tokens: int = 6144,
) -> list[LongBenchSample]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    candidates = []
    for row in _read_rows(path):
        official_length = int(row["length"])
        if not min_official_length <= official_length < max_official_length:
            continue
        document_ids = list(tokenizer.encode(str(row["context"]), add_special_tokens=False))
        if not min_document_tokens <= len(document_ids) <= max_document_tokens:
            continue
        references = [str(value).strip() for value in row["answers"] if str(value).strip()]
        if not references:
            continue
        source_id = str(row["_id"])
        rank = hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()
        candidates.append((rank, row, document_ids, references))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < offset + count:
        raise ValueError(
            f"Only {len(candidates)} eligible LongBench rows, "
            f"need {offset + count} for offset={offset}, count={count}"
        )

    samples = []
    for _, row, document_ids, references in candidates[offset : offset + count]:
        source_id = str(row["_id"])
        token_blocks = [
            document_ids[index : index + block_size]
            for index in range(0, len(document_ids), block_size)
        ]
        samples.append(
            LongBenchSample(
                sample_id=f"longbench-2wiki-{source_id}",
                story_id=f"longbench-2wiki-{source_id}",
                blocks=[f"PRETOKENIZED_BLOCK_{index:04d}" for index in range(len(token_blocks))],
                question=str(row["input"]).strip(),
                answer=references[0],
                answer_aliases=references,
                metadata={
                    "benchmark": "LongBench-E",
                    "task": "2wikimqa_e",
                    "source_id": source_id,
                    "source_context": str(row["context"]),
                    "references": references,
                    "official_length_words": int(row["length"]),
                    "document_tokens": len(document_ids),
                    "block_token_ids": token_blocks,
                    "counterfactual_group": source_id,
                    "evidence_order": "not_available",
                },
            )
        )
    return samples


def selection_manifest(samples: list[LongBenchSample]) -> dict[str, Any]:
    return {
        "samples": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "document_tokens": [int(sample.metadata["document_tokens"]) for sample in samples],
        "block_counts": [len(sample.blocks) for sample in samples],
    }
