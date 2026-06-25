"""Prompt templates."""

from __future__ import annotations

import json

from pearl.data.schema import PackedChunk


def build_qa_prompt(question: str, chunks: list[PackedChunk]) -> str:
    """Build a grounded QA prompt."""

    evidence = "\n\n".join(f"[{idx}] {chunk.text}" for idx, chunk in enumerate(chunks, start=1))
    return (
        "You are given a question and a set of evidence excerpts.\n"
        "Answer using only the evidence.\n"
        "If the evidence is insufficient, say that the answer is not supported.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer:"
    )


def build_citation_prompt(question: str, chunks: list[PackedChunk]) -> str:
    """Build a prompt that asks for JSON answer and evidence ids."""

    evidence = "\n".join(f"[{idx}] {chunk.text}" for idx, chunk in enumerate(chunks, start=1))
    schema = json.dumps({"answer": "...", "supporting_evidence_ids": ["..."]}, indent=2)
    return (
        "Answer the question using only the evidence.\n"
        "After the answer, cite the evidence IDs that support it.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Return JSON:\n{schema}"
    )
