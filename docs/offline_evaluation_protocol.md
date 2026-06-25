# Offline Logged-Candidate Evaluation Protocol

Milestone 2 evaluates rankers within the candidate sets already present in logged MIND
impressions. It does not retrieve candidates from the full news catalog.

## Chronological Validation

The Milestone 1 pipeline sorts official MIND train behaviors by timestamp and creates a
chronological train/validation split. Baselines are fit on the train partition and evaluated
on validation. This avoids random row-level splitting, which can leak future user behavior
and news popularity patterns into earlier examples.

## Parameter Selection

Validation metrics select hyperparameters such as the time-decay half-life. Test labels are
not inspected during this step. The default time-decay grid is 6, 24, and 72 hours, using:

```text
weight = 0.5 ** (age_hours / half_life_hours)
```

The age is measured from observed click-event timestamps relative to the fitting cutoff. It
is not article age; MIND publication time is not available in the processed schema.

## Final Test Evaluation

After validation selection, selected configurations are refit on train plus validation when
they use fitted click statistics or a fitted text vocabulary. They are then evaluated once on
the official dev-derived test partition. Test metrics are kept separate from validation
metrics and are never used for model or hyperparameter selection.

## Impression-Grouped Ranking

Metrics are computed inside each impression group. A model receives only the row-supplied
history and the logged candidate set for that impression. The clicked candidates in the
current row are not appended to the history before scoring the row.

Score ties are deterministic:

1. higher model score
2. lower original candidate position

This preserves source order as a stable tie-breaker and avoids nondeterministic row shuffles.

## Metric Handling

The implementation reports MRR, NDCG@5, NDCG@10, Recall@5, Recall@10, Hit Rate@5,
Hit Rate@10, and impression-level AUC.

Labeled impressions with no clicked candidate receive zero for MRR, NDCG, recall, and hit
rate. They are counted rather than hidden. AUC is skipped when an impression has only one
class, and the skip count is reported. Unlabeled inference-style impressions are skipped for
metric computation and counted explicitly.

## Leakage Safeguards

- Validation models fit only train rows.
- Final test models fit train plus validation rows only after hyperparameters are fixed.
- Candidate labels are never used by category-affinity or TF-IDF history profiles.
- Popularity baselines count positive candidate clicks only in the allowed fitting partition.
- TF-IDF vocabulary is fit only on articles referenced by allowed fitting rows.
- Prediction outputs are checked for one row per original candidate per baseline.
- Original candidate order, labels, and impression grouping are preserved.

The TF-IDF baseline separates label leakage from vocabulary leakage. It does not use
validation or test labels, and its final vocabulary is refit only after validation selection.

## Limits Of Interpretation

Logged-candidate metrics are useful for comparing rankers over the candidates exposed by a
previous system. They do not estimate full-catalog retrieval quality. Offline click metrics
also inherit exposure bias: users clicked only what they were shown, and missing clicks are
not the same as negative editorial relevance.

Future retrieval milestones should evaluate candidate generation separately.

