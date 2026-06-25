"""Run a tiny PEARL smoke experiment.

This module is intentionally local and lightweight. It verifies that the
research pipeline can build a context package before heavyweight models or
datasets are connected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl.data.schema import Document, EvidenceSpan, LongContextExample
from pearl.evaluation.evidence_metrics import evidence_recall
from pearl.packing.pack_builder import pearl_pack
from pearl.utils.io import read_yaml, write_json
from pearl.utils.seeding import set_seed


def build_smoke_example() -> LongContextExample:
    """Create a deterministic toy long-context example."""

    text = (
        "The report opens with background on long-context language models. "
        "Several distractor paragraphs discuss training hardware and unrelated datasets. "
        "PEARL improves long-context QA by selecting evidence, compressing it, and placing "
        "important facts near prompt edges. "
        "Another paragraph mentions that graph coverage helps preserve bridge evidence for "
        "multi-hop questions. "
        "The conclusion repeats that position-aware ordering reduces lost-in-the-middle risk."
    )
    start = text.index("PEARL improves")
    end = text.index("Another paragraph") - 1
    return LongContextExample(
        example_id="smoke_001",
        dataset="smoke",
        question="How does PEARL improve long-context QA?",
        documents=[Document(doc_id="doc_0", title="PEARL Smoke", text=text)],
        answer="by selecting evidence, compressing it, and placing important facts near prompt edges",
        gold_evidence=[
            EvidenceSpan(doc_id="doc_0", char_start=start, char_end=end, text=text[start:end])
        ],
        task_type="single_hop",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/metrics/smoke_pearl.json")
    args = parser.parse_args()

    config = read_yaml(args.config) if Path(args.config).exists() else {}
    set_seed(int(config.get("seed", 42)))
    example = build_smoke_example()
    prompt, packed = pearl_pack(example, config=config)
    result = {
        "example_id": example.example_id,
        "method": packed.method,
        "num_chunks": len(packed.chunks),
        "total_tokens": packed.total_tokens,
        "compression_ratio": packed.compression_ratio,
        "evidence_recall": evidence_recall(packed.chunks, example.gold_evidence),
        "prompt_preview": prompt[:500],
    }
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
