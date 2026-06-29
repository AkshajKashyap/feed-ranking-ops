# Model Selection And Local Serving

## Purpose

Milestone 6 turns offline logged-candidate results into an explicit serving decision. It
compares the strongest simple baseline with the learned ranker, applies a documented
validation-only promotion threshold, packages the selected policy, and serves it through a
small FastAPI application.

The service ranks only candidate news IDs supplied by the caller. It is not a full-catalog
retrieval service.

## Promotion Rule

The default rule is:

1. Find the strongest baseline by validation NDCG@10.
2. Use validation MRR, validation AUC, then policy name as deterministic tie-breakers.
3. Compare the validation-selected learned ranker with that baseline.
4. Promote the learned ranker only if its relative validation NDCG@10 improvement is at
   least 3%.
5. Otherwise retain the simpler baseline policy.

The threshold is configurable with `--minimum-relative-improvement`. Three percent is
deliberately conservative: a small offline gain must be large enough to justify a more complex
serving artifact. The current HGB ranker improved validation NDCG@10 by about 2.77%, so it
does not clear the default threshold and category affinity remains selected.

Internal-test metrics appear in the report only after the decision. They never select or
reject a policy. In the current real-data run, category affinity also slightly exceeded HGB
on internal-test NDCG@10, while HGB improved Recall@10 and Hit Rate@10. Both facts are
retained rather than converted into an unqualified model win.

## Packaged Artifact

Policy selection writes:

```text
reports/model_selection/
  promotion_report.json
  promotion_report.md

artifacts/serving/
  policy_manifest.json
  news_catalog.parquet
```

The manifest records validation and post-selection test metrics, the threshold and observed
lift, data protocol, fitting partitions, Git commit, an internal-holdout warning, policy
configuration, limitations, relative artifact paths, request schema version, and expected
catalog columns.

The category-affinity serving artifact contains only article IDs, categories, and
subcategories. It contains no click labels or user histories. Startup validates the Pydantic
manifest, ensures artifact paths stay inside the manifest directory, and verifies the catalog
schema before accepting requests.

Generated reports and artifacts remain ignored by Git.

## Why The Baseline Is Served

Category affinity is stateless and its online score exactly matches the established offline
baseline:

```text
score = category_weight * history_category_count
      + subcategory_weight * history_subcategory_count
```

The learned HGB ranker is not served in this milestone. Milestone 5 did not persist a
self-contained model plus TF-IDF/preprocessing state, and reconstructing a partial or
different feature path online would violate offline/online consistency. If a lower threshold
selects HGB, the manifest records the promotion but marks `serving_ready` false and the API
fails closed.

## Commands

Select and package the policy:

```bash
python -m feed_ranking_ops.ranking.select_policy \
  --baseline-reports-dir reports/baselines \
  --ltr-reports-dir reports/ltr \
  --processed-dir data/processed \
  --reports-dir reports/model_selection \
  --artifacts-dir artifacts/serving

make select-policy
```

Run a local in-process endpoint smoke:

```bash
make smoke-serve
```

Start the API:

```bash
uvicorn feed_ranking_ops.serving.app:app \
  --host 127.0.0.1 \
  --port 8000

make serve
```

Set `FEED_RANKING_POLICY_MANIFEST` to use a manifest outside the default
`artifacts/serving/policy_manifest.json`.

## Endpoints

### `GET /health`

Returns whether the manifest and catalog loaded successfully. A missing artifact yields a
degraded health response; policy and ranking requests return HTTP 503.

### `GET /policy`

Returns public promotion metadata, metrics, fitting partitions, protocol, and limitations.
Internal artifact paths are not exposed.

### `POST /rank`

Request:

```json
{
  "history_news_ids": ["N1", "N2"],
  "candidate_news_ids": ["N3", "N4", "N5"],
  "timestamp": "2019-11-15T12:00:00Z"
}
```

Response:

```json
{
  "selected_policy": "category_affinity",
  "policy_family": "category_affinity",
  "ranked_candidates": [
    {
      "news_id": "N4",
      "original_position": 1,
      "rank": 1,
      "score": 1.5
    }
  ],
  "missing_candidate_ids": [],
  "unknown_history_ids": [],
  "metadata": {
    "tie_breaker": "original_candidate_position"
  },
  "warnings": []
}
```

Unknown history IDs are excluded from the profile and returned explicitly. Unknown
candidate IDs are omitted from ranking and returned explicitly. Empty histories use fallback
scores, preserving caller candidate order as the deterministic tie-breaker.

## Limitations

- Rankings are confined to caller-supplied candidates and inherit logged-exposure bias.
- The internal test is a chronological holdout from MIND-small train, not an official MIND
  benchmark result.
- Category affinity is a simple behavioral heuristic, not a semantic relevance model.
- This local service has no authentication, rate limiting, monitoring, or online
  experimentation.
- Learned-ranker serving remains deferred until its complete fitted state can be packaged
  and schema-validated.
