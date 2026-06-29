# Experimental Methodology

## Research Questions

The repository separates three questions:

1. Can simple or learned policies reorder a logged candidate set?
2. Can history-based retrieval find clicked articles in an eligible catalog?
3. Does an approximate index reproduce a chosen dense retrieval representation efficiently?

Metrics across these questions are not interchangeable.

## Chronological Splitting

Recommendation data changes over time. Random splitting would expose later article and click
patterns to earlier examples. The train-only protocol stable-sorts official training
behaviors by timestamp, source row, and impression ID, then assigns 70% train, 15%
validation, and 15% internal test.

Histories remain exactly as supplied by MIND. Boundary timestamps, row hashes, labels, and
partition-overlap checks are recorded in split metadata.

## Validation Selection And Internal Test

Every configurable experiment follows:

1. fit on chronological train;
2. compare configurations on validation;
3. select using a declared validation metric;
4. refit the selected configuration on train plus validation;
5. evaluate once on internal test.

Internal-test metrics never select configurations or policies. They are post-selection
estimates and are explicitly labeled as a train-only chronological holdout.

## Logged-Candidate Ranking

Logged-candidate evaluation ranks only the articles in each MIND impression. Metrics include
MRR, NDCG@5/10, Recall@5/10, Hit Rate@5/10, and impression AUC. Equal scores use original
candidate position as deterministic tie-breaker.

Pointwise LTR treats each candidate click as a binary target and uses predicted probability
as a ranking score. The surrogate classification objective does not optimize NDCG directly.

## Exact Retrieval And ANN

Sparse exact retrieval is a correctness-oriented full-catalog path with explicit observed
availability, history exclusion, and popularity fallback. It is intentionally bounded in
real-data smoke runs because scoring many catalog vectors per query is expensive.

ANN benchmarking separates:

- representation loss: sparse TF-IDF versus SVD dense vectors;
- search loss: dense exact versus FAISS;
- operating cost: index build, batched search, filtering, memory, and total runtime.

Fast ANN-only mode skips sparse/dense exact references. It still reports clicked-target
retrieval quality but cannot claim approximation recall for that run.

## Promotion Threshold

The strongest baseline and selected learned ranker are compared on validation NDCG@10.
Validation MRR, AUC, and policy name are deterministic tie-breakers. A learned policy must
improve validation NDCG@10 by at least 3% relative before its additional complexity is
accepted.

HGB improved validation NDCG@10 by 2.77%, so it was rejected. Category affinity remained
selected. The later internal-test result, where category affinity was slightly stronger on
NDCG@10, is evidence reported after the decision rather than a selection input.

## Negative Results

Weak retrieval metrics, high category-policy tie rates, unavailable ANN approximation recall
in fast mode, and the failed HGB promotion are preserved. A portfolio result is credible only
if disappointing evidence changes the system decision rather than disappearing from the
report.

## Reproducibility

Configuration, seeds, fitting partitions, timing, environment information, metric
definitions, selected configuration, and test-label isolation are stored in JSON reports.
Small deterministic portfolio summaries are tracked; raw data and large generated artifacts
are not.
