# FeedRank Ops Portfolio Summary

A leakage-aware Microsoft MIND-small recommendation project covering chronological preparation, retrieval, ranking, promotion, serving, and offline monitoring.

> Internal-test values use a train-only chronological holdout and are not official MIND validation results.

## Results At A Glance

| Component | Result |
| --- | --- |
| Best logged-candidate baseline | `category_affinity`; validation NDCG@10 0.3552, internal-test 0.3377 |
| Pointwise LTR | `hist_gradient_boosting_learning_rate-0.1_max_leaf_nodes-15`; validation NDCG@10 0.3651, internal-test 0.3351 |
| Exact retrieval smoke | 200 queries in 101.1838 s |
| Batched FAISS ANN | 2000 queries in 41.3185 s |
| Selected serving policy | category_affinity |
| Monitoring | 3 logged smoke requests; validation tie fraction 92.84% |

## Data And Protocol

- Dataset: Microsoft MIND-small
- Protocol: train_only_chronological
- Split rows: {'test': 23545, 'train': 109875, 'validation': 23545}

## Promotion Decision

- Decision: promote_baseline_policy
- Learned ranker: rejected_insufficient_improvement
- Observed validation lift: 2.77%
- Required lift: 3.00%

## Serving

- Endpoints: /health, /policy, /rank, /metrics
- Request logging: Optional aggregate JSONL; raw history and candidate IDs are not logged.

## Limitations

- Internal-test metrics use a train-only chronological holdout, not official MIND validation.
- Logged-candidate metrics inherit exposure and position bias.
- Exact retrieval is a bounded correctness smoke, not a scalable production path.
- ANN-only runs without dense exact comparison do not report approximation recall.
- Category affinity produces frequent score ties and uses source position as tie-breaker.
- Serving telemetry and drift checks are local/offline diagnostics.

## Recommended Next Work

- Evaluate the protocol on an additional public temporal recommendation dataset.
- Package a future learned ranker only with complete fitted preprocessing state.
- Add hosted deployment only when a real operating environment exists.
