# Stage H: arbitrary pairs and evidence chains

## Why this stage exists

Stages V10-F2 and G1a established a clean query-aware block-compression protocol, but the learned interaction model was limited to adjacent block boundaries. G1b separated examples were therefore a negative control for that model, not evidence that it could retrieve distant dependencies. On RULER Variable Tracking, the old pair model matched its shuffled control and did not improve over the matched unary selector.

Stage H does not replace those results. It tests the narrower follow-up question:

> Are aggregate features of arbitrary block pairs sufficient for selecting distributed evidence chains, or is explicit graph connectivity required?

## Frozen design

### Controlled chains

- 36 stories from 18 counterfactual families.
- 42 blocks of 96 tokens, approximately 4k document tokens.
- Fixed budget: 6 blocks.
- Chain lengths: 2, 3 and 4 evidence blocks.
- Placements: adjacent and dispersed.
- Three families in every chain-length/placement stratum.
- Counterfactual partners keep positions and relation schema fixed while changing the correct chain.

The target laboratory is mentioned exactly once. Distractors contain complete alternative chains, so retaining one obvious endpoint is insufficient.

### External transfer

- 20 examples from official NVIDIA RULER Variable Tracking.
- Length: 4096 tokens.
- Pinned RULER revision: `c3f5e3b`.
- Fixed budget: 6 blocks.

This subset is an exploratory transfer check. The completed 50-example V16 RULER runs remain the primary external benchmark results.

## Methods

1. `graph_matched_unary_ridge_v1`: Ridge using only unary block aggregates.
2. `graph_arbitrary_pair_ridge_v1`: the same Ridge target, folds, candidate pool and frozen `alpha`, with arbitrary-pair aggregates added.
3. `graph_pair_shuffled_v1`: unary features unchanged; the full pair correspondence between blocks is permuted.
4. `relation_graph`: training-free lexical graph propagation. This is a mechanism diagnostic, not the learned BIKV model.
5. `snapkv_upstream_query_aware`: matched-budget external baseline in its native query-in-prefill protocol.
6. `full` and, for controlled chains, `task_oracle`.

The learned primary comparison is pair Ridge versus matched unary Ridge. The relation graph must be reported separately.

## Candidate construction

Query attention, KNorm and a training-free relation score form a shortlist of up to 12 blocks. Every pair in the shortlist can anchor a candidate. The remaining budget is filled by either unary ranking or graph expansion. Both learned models score exactly the same masks.

Distance and adjacency indicators are excluded from the primary learned model. They cannot reveal which placement condition generated a story.

## CPU findings before GPU

The frozen arbitrary-pair Ridge was audited on the existing V10 causal utility labels:

| Model | Support retention | Utility regret |
|---|---:|---:|
| Matched unary Ridge | 73.3% | 3.257 |
| Arbitrary-pair Ridge | 73.3% | 3.239 |

Pair-minus-unary retention was 0.0 percentage points. Utility-regret reduction was 0.018 with a family-bootstrap interval crossing zero. Thus the old labels do not establish a learned arbitrary-pair benefit.

As a structural sanity check, the training-free relation graph retained every support block in all 360 generated test stories used by the CPU audit. This proves only that the generated chains are recoverable from their relation structure. It does not establish LLM answer accuracy.

## Interpretation rules

- Strong outcome: pair Ridge beats matched unary and shuffled pair on fresh controlled families, with a positive family-bootstrap interval, and transfers at least directionally to RULER.
- Mechanistic outcome: relation graph succeeds but pair Ridge does not. The conclusion is that aggregate second-order features are insufficient and explicit chain connectivity is needed.
- Null outcome: neither learned pairs nor relation graph improve answer quality. The adjacent-boundary result remains valid only for localized two-block evidence.

No result from Stage H should be merged numerically with V16 RULER 8k, whose full-context backbone achieved only 34.7% Set F1.

## Compute plan

Run the two notebooks independently and, if resources permit, in parallel on separate accounts:

- `BIKV_V17_Stage_H_Chains_4k_Colab_Kaggle.ipynb`: expected 2 to 3.5 hours on one T4.
- `BIKV_V17_Stage_H_RULER_4k_Subset_Colab_Kaggle.ipynb`: expected 45 to 90 minutes on one T4.

Every result row is written atomically. Colab mirrors rows and rolling archives to Drive after every two samples. Kaggle writes rolling archives to `/kaggle/working`; a saved version preserves them.

No 2k or 8k Stage H run is preregistered. Those lengths are justified only if the controlled 4k experiment produces a non-null signal.
