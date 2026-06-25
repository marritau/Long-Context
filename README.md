# PEARL: Long-Context Processing in Large Language Models

Research project on long-context processing in large language models.

Working topic:

> Research and improvement of long-context usage in LLMs through query-aware
> compression and position-aware context packing.

The repository implements an MVP of **PEARL**:

**Position-aware Evidence Assembly for Reliable Long-context QA**.

PEARL treats long-context QA as an input-side optimization problem:

```text
question + documents + token budget
  -> chunking
  -> hybrid retrieval
  -> evidence graph
  -> MMR coverage selection
  -> query-aware compression
  -> position-aware ordering
  -> prompt
```

The goal is not to train a new long-context model. The goal is to study how
selection, compression, and placement of evidence affect answer quality,
evidence recall, token cost, and position sensitivity.

## Research Questions

1. How strongly does answer quality depend on the position of evidence inside a
   long prompt?
2. Under a fixed token budget, when is full context worse than retrieval,
   compression, or PEARL-style packing?
3. Does PEARL stay robust when lexical overlap between question and evidence is
   low?
4. Does graph-based evidence coverage help multi-hop QA compared with plain
   top-k retrieval?
5. Can the same findings be transferred to Russian long-context tasks?

## Repository Layout

```text
configs/          YAML configs for datasets, models, and experiments
data/             raw, processed, index, and sample data folders
src/pearl/        PEARL package
scripts/          CLI helper scripts
tests/            unit tests for the MVP core
report/           LaTeX report skeleton
results/          predictions, metrics, tables, and figures
notebooks/        analysis notebooks placeholder
```

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m pytest
```

Run a tiny local smoke experiment:

```bash
python -m pearl.experiments.run_experiment --config configs/default.yaml
```

The current implementation intentionally keeps the core runnable without model
downloads. Dense retrieval and reranking have deterministic lightweight
fallbacks; heavier backends such as SentenceTransformers, FlagEmbedding, vLLM,
LLMLingua, and LongLLMLingua can be connected behind the wrapper modules.

## Main Methods

Implemented MVP components:

- structure-aware chunking with token overlap;
- BM25 lexical retrieval;
- deterministic hashing-based dense retrieval fallback;
- reciprocal rank fusion;
- lightweight cross-encoder-style rerank fallback;
- evidence graph with semantic, adjacency, and section edges;
- personalized PageRank over evidence chunks;
- MMR selection with redundancy penalty;
- sentence-level query-aware compression with neighbor preservation;
- position-aware context ordering using a lost-in-the-middle utility curve;
- answer, evidence, cost, and position metrics.

Planned heavyweight baselines:

- LongLLMLingua;
- Selective Context;
- BGE embeddings and reranker;
- vLLM/Hugging Face generation runners.

## MVP Experiment Plan

The recommended first milestone:

```text
Datasets: RULER + NoLiMa
Model: Qwen2.5-7B-Instruct or Phi-3-mini-128k
Methods: full context, first-B, BM25, dense, hybrid rerank, PEARL-full
Metrics: answer accuracy, evidence recall, token count, PSI, middle drop
```

Then add Qasper, ablations, budget sweep, and optional LIBRA/Russian evaluation.
