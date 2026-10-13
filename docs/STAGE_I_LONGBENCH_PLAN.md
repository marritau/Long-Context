# Stage I: LongBench-E 2Wiki natural-text transfer

Stage I is a frozen exploratory transfer experiment on natural Wikipedia passages. It is independent from Stage H controlled chains and can run concurrently in another Colab runtime.

## Frozen cohort

- Official dataset: `zai-org/LongBench`, configuration `2wikimqa_e`.
- Pinned revision: `0962550175777a5b10df401ca715e01c0bb3ee8d`.
- Cohort is defined by Qwen token length, not LongBench's word-count bins.
- Qwen document length filter: 3072 to 6144 tokens.
- Selection: 24 examples ranked only by SHA-256 of seed and source id.
- Seed: `20261019`.
- Block size: 96 tokens.
- Fixed memory budget: six blocks.

The resulting frozen cohort contains documents with 3100 to 6019 tokens and 33 to 63 blocks. The selected ids and token lengths are written before evaluation and checked on resume.

## Methods

- full context;
- matched unary Ridge;
- arbitrary-pair Ridge;
- shuffled-pair control;
- relation graph;
- matched-budget upstream SnapKV.

All learned weights are frozen before LongBench is read. No LongBench example is used for training or hyperparameter selection.

## Metrics

Primary metric: official normalized token-level answer F1, maximized across references. Secondary metric: normalized exact match. LongBench does not expose block-level supporting evidence in its standardized records, so support retention is intentionally not reported.

## Interpretation

This 24-example run is exploratory. A pair-over-unary advantage is meaningful only if its paired bootstrap interval is positive and the real pair model also exceeds the shuffled control. The result complements, rather than replaces, the completed 50-example RULER evaluations.
