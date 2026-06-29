# Model And System Card

## Overview

| Field | Value |
| --- | --- |
| System | FeedRank Ops |
| Version | 0.1.0 |
| Dataset | Microsoft MIND-small |
| Task | Personalized logged-candidate news ranking |
| Selected policy | Category affinity |
| Learned candidate | Pointwise HistGradientBoostingClassifier |
| Serving decision | Learned candidate not promoted |

The deployed local policy scores a candidate from category and subcategory counts in the
caller's supplied history:

```text
score = 1.0 * category_history_count
      + 0.5 * subcategory_history_count
```

Source candidate position breaks score ties.

## Intended Use

- Education and portfolio demonstration of recommendation-system evaluation and operations.
- Reproducible offline experimentation on MIND-small.
- Local ranking of caller-supplied candidate sets using a packaged policy.
- Study of temporal splitting, leakage controls, promotion decisions, and observability.

## Out-Of-Scope Use

- Editorial or safety-critical news decisions.
- Political, medical, financial, or emergency-information prioritization.
- Autonomous content moderation or user profiling.
- Production deployment without privacy, security, load, and fairness review.
- Full-feed retrieval through the `/rank` endpoint.

## Data And Target

MIND click labels are implicit behavioral feedback. They reflect what the original system
exposed and what a user clicked, not an unbiased relevance judgment. News text, category,
subcategory, row histories, timestamps, and logged candidate sets are used.

The verified release uses a train-only chronological split:

- train: 109,875 impressions;
- validation: 23,545 impressions;
- internal test: 23,545 impressions.

The internal test is not official MIND validation or test data, and its metrics are not
directly comparable with official benchmark results.

## Evaluation

| Policy | Validation NDCG@10 | Internal-test NDCG@10 | Validation MRR | Internal-test MRR |
| --- | ---: | ---: | ---: | ---: |
| Category affinity | 0.3552 | 0.3377 | 0.3179 | 0.3008 |
| HGB LTR | 0.3651 | 0.3351 | 0.3273 | 0.2965 |

HGB improved validation NDCG@10 by 2.77%, below the predeclared 3% relative promotion
threshold. Category affinity was selected without consulting internal-test metrics. HGB's
stronger Recall@10 and Hit Rate@10 are retained in reports, as is its weaker internal-test
NDCG@10.

## Leakage Controls

- Behavior partitions are chronological, never random.
- Validation fitting uses train only.
- Final test fitting uses train plus validation only after configuration selection.
- Test labels never choose a baseline, retrieval configuration, ranker, or serving policy.
- Training popularity features use only prior impressions and exclude the current label.
- Same-row clicks are not appended to the supplied user history.
- Text vocabularies follow explicit fitting-partition policies.

## Serving Artifact

The Pydantic manifest records promotion evidence, schema versions, policy configuration,
fitting partitions, Git revision, data protocol, and limitations. The companion catalog
contains only ID/category/subcategory metadata. Startup rejects missing, malformed, escaping,
or schema-inconsistent artifacts.

## Observability

Rank logging is optional. Events contain counts, latency, outcome, warnings, and category
aggregates but no raw history or candidate IDs. In-process metrics reset on restart. Offline
monitoring reports policy slices and distribution comparisons; it is not continuous
production monitoring.

## Limitations And Risks

- Click labels contain exposure, position, and selection bias.
- Category affinity cannot model semantics within the same category.
- Score ties occur in 92.84% of validation and 94.10% of internal-test impressions.
- The service's tie-breaker can preserve bias from caller candidate order.
- Exact retrieval is computationally expensive.
- The 1,000-query-per-partition ANN smoke skipped dense exact comparison.
- Small request logs are not representative drift samples.
- No fairness analysis, authentication, rate limiting, or hosted monitoring is included.

## Privacy

MIND is a public research dataset governed by its own terms. Raw and processed data remain
outside Git. Local request logs are development artifacts and should follow environment
retention and access rules even though raw IDs are omitted.
