# Logged-Candidate Learning-to-Rank Protocol

## Scope

Milestone 5 trains pointwise supervised rankers over the candidate articles already present
in each MIND impression. Each candidate becomes one training row, its click label is the
binary target, and its predicted click probability is used as a ranking score within that
impression.

This is a second-stage logged-candidate experiment. It does not retrieve from the complete
catalog, measure serving behavior, or constitute an official MIND test submission.

## Models

The deterministic validation grid contains:

- balanced logistic regression with standardized numeric features;
- balanced `HistGradientBoostingClassifier` models with small learning-rate settings.

Validation NDCG@10 selects the configuration, with MRR and model name as deterministic
tie-breakers. The selected configuration is rebuilt and refit before final evaluation.

## Features

All features are numeric and inspectable:

| Group | Features |
| --- | --- |
| Source order | candidate position and reciprocal position |
| Prior engagement | global and 24-hour time-decayed click counts |
| History affinity | category count, subcategory count, and TF-IDF similarity |
| Article metadata | category/subcategory frequency, abstract presence, title and abstract length |
| History quality | total history length, known history count, and empty-history indicator |
| Repetition | candidate-seen-in-history indicator |

MIND's processed article schema has no trustworthy publication timestamp, so the ranker does
not manufacture an article-age feature. Category frequencies and the TF-IDF vocabulary use
only article metadata referenced by the allowed fitting partitions.

## Leakage Controls

Validation uses train-fitted artifacts only. Final test uses artifacts refit on train plus
validation. Test labels are never used for model selection or feature fitting.

Popularity features deserve an additional guard. For training rows, impressions are processed
chronologically and each impression is featurized before its labels update the click-count
state. The row therefore cannot include its own target or a future click in a popularity
feature. During validation and test, click-count snapshots contain only the explicitly allowed
earlier fitting partitions.

Current-impression labels never enter history, affinity, text, or metadata features. The
text-vectorizer policy matches the established baseline policy: fit vocabulary on articles
referenced by allowed fitting histories or candidates, without using their labels.

## Evaluation

1. Fit feature artifacts and every ranker on chronological train.
2. Evaluate all configurations and baselines on chronological validation.
3. Select by validation NDCG@10, then MRR.
4. Refit the selected configuration and feature artifacts on train plus validation.
5. Evaluate once on the internal chronological test partition.

MRR, NDCG@5/10, Recall@5/10, Hit Rate@5/10, and impression-level AUC use the same definitions
and deterministic source-position tie-breaking as the baseline pipeline. Reports compare the
learned models with original order, global popularity, time-decayed popularity, category
affinity, and TF-IDF similarity.

For the train-only preparation protocol, `test_behaviors.parquet` is an internal chronological
holdout. Its values are not directly comparable with official MIND validation results.

## Commands

Run a deterministic real-data smoke experiment, limiting each partition to 1,000 impressions:

```bash
python -m feed_ranking_ops.ranking.run_ltr \
  --processed-dir data/processed \
  --reports-dir reports/ltr_smoke \
  --limit-impressions 1000
```

Run the full experiment:

```bash
python -m feed_ranking_ops.ranking.run_ltr \
  --processed-dir data/processed \
  --reports-dir reports/ltr
```

The protocol JSON records fitting partitions, feature policy, selected configuration, stage
timings, and peak process memory. Prediction Parquet files retain impression boundaries,
source candidate positions, labels, scores, and deterministic predicted ranks.

## Interpretation

Improvement over the strongest baseline shows that the feature combination is useful within
logged candidate sets. It does not establish causal relevance, remove exposure bias, or imply
that full-catalog retrieval improved. Logistic coefficients and any available model
diagnostics describe fitted associations, not causal feature effects.
