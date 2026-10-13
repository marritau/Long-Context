# Stage I-B: confirmatory LongBench-E 2Wiki replication

Stage I-B is the only planned GPU run after Stage I. It repeats the frozen natural-text
transfer protocol on a disjoint holdout cohort; it does not tune or replace any selector.

## Frozen design

- Dataset revision, tokenizer length filter, seed, block size and six-block budget are unchanged.
- The 87 eligible examples are ranked exactly as in Stage I.
- Stage I used ranks 1-24; Stage I-B uses ranks 25-48 (`selection_offset=24`).
- A checked-in Stage I manifest makes overlap a fatal error.
- All six methods and the Stage H frozen weights are unchanged.
- The complete selected natural-text cohort is stored in the artifact bundle.

## Predeclared comparisons

1. arbitrary-pair Ridge minus matched-unary Ridge answer F1;
2. arbitrary-pair Ridge minus shuffled-pair control answer F1;
3. arbitrary-pair Ridge minus matched-budget SnapKV answer F1;
4. pooled Stage I + Stage I-B estimates on 48 independent examples.

Stage I-B is confirmatory replication. No model, feature, threshold or cohort criterion may be
changed after its predictions are inspected. After completion, only offline pooled analysis remains.
