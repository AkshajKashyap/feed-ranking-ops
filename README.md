FeedRank Ops
============

A two-stage personalized news-feed recommendation system covering temporal data
preparation, candidate retrieval, ranking, serving, feedback simulation, and monitoring.

Current Scope
-------------

Milestone 1 builds the data foundation for Microsoft MIND-small:

- Parse and validate `news.tsv` and `behaviors.tsv`.
- Audit source data quality and coverage.
- Prepare deterministic chronological splits using either official train/dev sources or an
  explicit train-only internal holdout protocol.
- Write inspectable Parquet outputs and split metadata.

Milestone 2 adds offline logged-candidate ranking evaluation:

- Load and validate processed Parquet outputs.
- Explode nested behavior rows into inspectable candidate-level rows.
- Evaluate MRR, NDCG, recall, hit rate, and impression-level AUC.
- Compare original-order, popularity, time-decayed popularity, category-affinity, and
  TF-IDF content-similarity baselines.
- Tune time-decay half-life on validation only, then refit and evaluate once on test.

Milestone 3 adds exact full-catalog candidate retrieval:

- Build retrieval queries from behavior histories and clicked impression targets.
- Derive observed article availability from first candidate appearance timestamps.
- Fit sparse TF-IDF article vectors under an inductive fitting protocol.
- Build mean or recency-weighted history profiles.
- Retrieve top-K articles exactly with cosine similarity from the eligible catalog.
- Use popularity fallback for empty or unusable histories.
- Select retrieval configuration on validation Recall@100 and evaluate once on test.

Milestone 4 adds dense-vector and FAISS approximate retrieval benchmarking:

- Project sparse TF-IDF article vectors into deterministic dense vectors with TruncatedSVD.
- Evaluate dense exact inner-product retrieval as the correctness reference for FAISS.
- Compare FAISS Flat IP and HNSW IP against dense exact retrieval.
- Select SVD dimension and HNSW search parameters on validation-only ANN agreement.
- Separate representation loss from ANN search loss in the generated reports.

Milestone 5 adds supervised learning-to-rank over logged candidates:

- Explode impression candidates into deterministic pointwise training rows.
- Combine source order, prior engagement, history affinity, TF-IDF, and article metadata.
- Compare balanced logistic regression and histogram gradient boosting.
- Select by validation NDCG@10, refit on train plus validation, and evaluate once on the
  internal chronological holdout.
- Prevent same-impression and future-click leakage in popularity features.

Milestone 6 adds validation-only policy promotion and local serving:

- Compare the strongest baseline with the validation-selected learned ranker.
- Require a configurable minimum relative NDCG@10 improvement before adding complexity.
- Retain category affinity when HGB does not clear the default 3% validation threshold.
- Package a versioned manifest and compact article metadata artifact.
- Serve deterministic caller-provided candidate ranking through FastAPI.

Milestone 7 adds serving observability and offline policy monitoring:

- Optionally append privacy-conscious JSONL telemetry for rank requests.
- Track process-local latency, error, missing-ID, and empty-history counters.
- Compute validation/internal-test metrics across policy behavior slices.
- Compare training reference distributions with validation, test, and request logs.
- Generate deterministic JSON and Markdown monitoring reports.

This scope intentionally does not implement neural rankers, two-tower models, LightGBM,
Redis, streaming, Docker, dashboards, hosted monitoring, or online experiments.

Expected Dataset Layout
-----------------------

The default `official_train_dev` protocol requires both official source directories:

```text
data/raw/
  MINDsmall_train/
    news.tsv
    behaviors.tsv
  MINDsmall_dev/
    news.tsv
    behaviors.tsv
```

When the official dev archive is unavailable, the opt-in `train_only_chronological` protocol
requires only:

```text
data/raw/
  MINDsmall_train/
    news.tsv
    behaviors.tsv
```

Train-only mode splits the official training behavior rows chronologically into 70% train,
15% validation, and 15% internal test. That final partition is an internal chronological
holdout, not the official MIND validation or test benchmark, and its metrics are not directly
comparable to official MIND validation results. The pipeline never switches protocols based
on which files happen to exist.

The project does not download MIND during tests or normal commands.

Commands
--------

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install optional FAISS support for ANN retrieval:

```bash
python -m pip install -e ".[dev,ann]"
```

Core imports and non-ANN tests do not require FAISS. The ANN command exits with a clear
message if `faiss-cpu` is not installed.

Validate the expected local file layout:

```bash
python -m feed_ranking_ops.data.validate_layout \
  --data-dir data/raw \
  --protocol official_train_dev
make validate-data
```

Validate the train-only layout:

```bash
python -m feed_ranking_ops.data.validate_layout \
  --data-dir data/raw \
  --protocol train_only_chronological
make validate-data-train-only
```

Audit either source protocol:

```bash
python -m feed_ranking_ops.data.audit_dataset \
  --data-dir data/raw \
  --reports-dir reports \
  --protocol official_train_dev
make audit-data

python -m feed_ranking_ops.data.audit_dataset \
  --data-dir data/raw \
  --reports-dir reports \
  --protocol train_only_chronological
make audit-data-train-only
```

Prepare official train/dev chronological splits:

```bash
python -m feed_ranking_ops.data.prepare_dataset \
  --data-dir data/raw \
  --output-dir data/processed \
  --reports-dir reports \
  --protocol official_train_dev
make prepare-data
```

Prepare the train-only 70/15/15 split:

```bash
python -m feed_ranking_ops.data.prepare_dataset \
  --data-dir data/raw \
  --output-dir data/processed \
  --reports-dir reports \
  --protocol train_only_chronological
make prepare-data-train-only
```

Run local checks that do not require the real MIND dataset:

```bash
make check
```

Evaluate baselines on already processed data:

```bash
python -m feed_ranking_ops.evaluation.run_baselines \
  --processed-dir data/processed \
  --reports-dir reports/baselines
make evaluate-baselines
```

Run a clearly labeled smoke evaluation over a limited number of impressions:

```bash
python -m feed_ranking_ops.evaluation.run_baselines \
  --processed-dir data/processed \
  --reports-dir reports/baselines_smoke \
  --limit-impressions 100
make evaluate-baselines-smoke
```

Run exact full-catalog retrieval on already processed data:

```bash
python -m feed_ranking_ops.retrieval.run_exact_retrieval \
  --processed-dir data/processed \
  --reports-dir reports/retrieval \
  --catalog-protocol observed_available
make evaluate-retrieval
```

Run a clearly labeled retrieval smoke evaluation:

```bash
python -m feed_ranking_ops.retrieval.run_exact_retrieval \
  --processed-dir data/processed \
  --reports-dir reports/retrieval_smoke \
  --limit-queries 100 \
  --text-configs title,title_abstract \
  --history-lengths 10,all \
  --decay-values 0.5
make evaluate-retrieval-smoke
```

Benchmark dense-vector exact retrieval and FAISS ANN retrieval:

```bash
python -m feed_ranking_ops.retrieval.run_ann_benchmark \
  --processed-dir data/processed \
  --reports-dir reports/ann \
  --catalog-protocol observed_available
make evaluate-ann
```

Run a smaller FAISS smoke benchmark:

```bash
python -m feed_ranking_ops.retrieval.run_ann_benchmark \
  --processed-dir data/processed \
  --reports-dir reports/ann_smoke \
  --limit-queries 100 \
  --svd-dims 32,64 \
  --ef-search 64 \
  --oversampling 4
make evaluate-ann-smoke
```

Run one real-data FAISS configuration without sparse or dense exact references:

```bash
python -m feed_ranking_ops.retrieval.run_ann_benchmark \
  --processed-dir data/processed \
  --reports-dir reports/ann_smoke_fast \
  --limit-queries 100 \
  --top-k 100 \
  --svd-dims 64 \
  --faiss-threads 4 \
  --ann-only \
  --single-config \
  --query-batch-size 256
```

ANN-only mode defaults to title text, a full-history mean profile, and FAISS Flat. It reports
clicked-target retrieval quality and detailed stage timings, but deliberately does not report
ANN agreement or approximation recall because dense exact retrieval is skipped. The original
full sparse/dense/Flat/HNSW comparison remains available without `--ann-only`. Query profiles
are stacked and searched in FAISS batches; iterative oversampling preserves per-query
availability and history filtering while avoiding one FAISS call per query.

Train a pointwise logged-candidate ranker on a 1,000-impression-per-partition smoke slice:

```bash
python -m feed_ranking_ops.ranking.run_ltr \
  --processed-dir data/processed \
  --reports-dir reports/ltr_smoke \
  --limit-impressions 1000
make evaluate-ltr-smoke
```

Run the full logged-candidate experiment:

```bash
python -m feed_ranking_ops.ranking.run_ltr \
  --processed-dir data/processed \
  --reports-dir reports/ltr
make evaluate-ltr
```

Select and package the serving policy:

```bash
python -m feed_ranking_ops.ranking.select_policy \
  --baseline-reports-dir reports/baselines \
  --ltr-reports-dir reports/ltr \
  --processed-dir data/processed \
  --reports-dir reports/model_selection \
  --artifacts-dir artifacts/serving
make select-policy
```

Smoke-test and run the local API:

```bash
make smoke-serve
make serve
```

Run a logged serving smoke and generate monitoring reports:

```bash
make smoke-serve-logged
make smoke-monitor
make monitor
```

Generated Outputs
-----------------

Audit writes:

- `reports/data_audit.json`
- `reports/data_audit.md`

Preparation writes:

- `data/processed/news.parquet`
- `data/processed/train_behaviors.parquet`
- `data/processed/validation_behaviors.parquet`
- `data/processed/test_behaviors.parquet`
- `data/processed/split_metadata.json`
- `reports/split_summary.md`

Baseline evaluation writes:

- `reports/baselines/validation_metrics.json`
- `reports/baselines/test_metrics.json`
- `reports/baselines/model_comparison.md`
- `reports/baselines/protocol.json`
- `reports/baselines/validation_predictions.parquet`
- `reports/baselines/test_predictions.parquet`

Exact retrieval writes:

- `reports/retrieval/validation_metrics.json`
- `reports/retrieval/test_metrics.json`
- `reports/retrieval/config_sweep.csv`
- `reports/retrieval/model_comparison.md`
- `reports/retrieval/protocol.json`
- `reports/retrieval/availability_summary.json`
- `reports/retrieval/validation_retrievals.parquet`
- `reports/retrieval/test_retrievals.parquet`

ANN retrieval writes:

- `reports/ann/validation_representation_metrics.json`
- `reports/ann/validation_ann_metrics.json`
- `reports/ann/test_representation_metrics.json`
- `reports/ann/test_ann_metrics.json`
- `reports/ann/config_sweep.csv`
- `reports/ann/latency_benchmark.csv`
- `reports/ann/model_comparison.md`
- `reports/ann/protocol.json`
- `reports/ann/runtime_environment.json`
- `reports/ann/selected_configuration.json`
- `reports/ann/index_metadata.json`
- `reports/ann/validation_retrievals.parquet`
- `reports/ann/test_retrievals.parquet`
- `reports/ann/query_diagnostics.parquet`

Fast ANN-only mode uses the same inspectable retrieval/diagnostic schemas and writes
`validation_metrics.json` and `test_metrics.json` in place of the full comparison metric files.

Learning-to-rank writes:

- `reports/ltr/validation_metrics.json`
- `reports/ltr/test_metrics.json`
- `reports/ltr/protocol.json`
- `reports/ltr/model_comparison.md`
- `reports/ltr/feature_importance.md`
- `reports/ltr/validation_predictions.parquet`
- `reports/ltr/test_predictions.parquet`

Policy selection writes:

- `reports/model_selection/promotion_report.json`
- `reports/model_selection/promotion_report.md`
- `artifacts/serving/policy_manifest.json`
- `artifacts/serving/news_catalog.parquet`

Serving observability and monitoring write:

- `artifacts/logs/rank_requests.jsonl` when request logging is enabled
- `reports/monitoring/monitoring_report.json`
- `reports/monitoring/monitoring_report.md`

Processed data and generated reports are ignored by git by default.

Parsing Semantics
-----------------

News records include:

- `news_id`
- `category`
- `subcategory`
- `title`
- `abstract`
- `url`
- `title_entities`
- `abstract_entities`

Behavior records include:

- `impression_id`
- `user_id`
- UTC-normalized `timestamp`
- ordered `history` list
- ordered `impressions` list of `{position, news_id, clicked}`

Click labels are preserved as `1`, `0`, or `null` for inference-style unlabeled candidates.
Audit mode counts invalid timestamps and malformed impression tokens. Preparation uses strict
parsing and fails until those issues are fixed.

Chronological Splitting
-----------------------

Random splitting can leak future behavior patterns into validation examples for temporal
recommendation systems. This project provides two explicit protocols:

- `official_train_dev` is the default. It assigns the first 80% of chronologically sorted
  official train behaviors to train, the remaining 20% to validation, and official dev to
  the final offline test partition.
- `train_only_chronological` is opt-in. It assigns the first 70% of official train behavior
  rows to train, the next 15% to validation, and the final 15% to an internal chronological
  holdout written to `test_behaviors.parquet`.

Both protocols:

- sort by timestamp, source-row number, and impression ID;
- use source-row order as the deterministic tie-break when a timestamp crosses a boundary;
- never splits an individual impression row across partitions;
- preserve source histories, candidate order, and labels;
- use no random splitting;
- record timestamp ranges and integrity checks in `split_metadata.json`.

The pipeline reports the observed official dev timestamp range honestly. It does not assume
the official dev period occurs after every official train event unless the local data verifies it.
In train-only mode, metadata and reports explicitly label the final partition as
`internal_chronological_holdout` and warn against treating it as an official MIND benchmark.

See [`docs/data_split_protocols.md`](docs/data_split_protocols.md) for protocol details.

Parquet Schema
--------------

`news.parquet` stores one row per unique `news_id`, keeping the first occurrence encountered
from official train then official dev.

Behavior Parquet files store:

- scalar source fields: `source_split`, `source_row_number`, `source_row_hash`,
  `impression_id`, `user_id`, `timestamp`;
- `history` as `list<string>` preserving source order;
- `impressions` as `list<struct<position:int32, news_id:string, clicked:int8?>>`
  preserving candidate order and nullable click labels.

Logged-Candidate Ranking
------------------------

Milestone 2 ranks only the candidate articles already present in each logged MIND
impression. This is not candidate retrieval from the full news catalog, and it should not be
interpreted as full-feed ranking quality.

Implemented baselines:

- `original_order`: keeps logged candidate order as a diagnostic source-order baseline.
- `global_popularity`: counts positive candidate clicks from the allowed fitting partition;
  unseen articles receive fallback score 0.
- `time_decayed_popularity`: counts positive clicks with exponential decay by observed click
  event time, not article publication time.
- `category_affinity`: scores candidates by matching category/subcategory to the current
  row's history profile.
- `tfidf_content_similarity`: fits a scikit-learn TF-IDF vocabulary on fitting-protocol
  articles and scores candidates by cosine similarity to the row history profile.

Metric definitions:

- MRR: reciprocal rank of the first clicked candidate.
- NDCG@5 and NDCG@10: binary-click discounted gain with deterministic tie-breaking.
- Recall@5 and Recall@10: clicked candidates recovered in the top K.
- Hit Rate@5 and Hit Rate@10: whether at least one clicked candidate appears in the top K.
- AUC: impression-level pairwise AUC, skipped when both classes are not present.

Labeled impressions with no clicked candidate receive zero for non-AUC ranking metrics and
are counted. Unlabeled inference-style impressions are skipped and reported.

Validation protocol:

1. Fit every baseline using chronological train only.
2. Evaluate on chronological validation.
3. Select hyperparameters, such as time-decay half-life, using validation metrics only.
4. Refit selected configurations on train plus validation when they have fitted statistics
   or vocabulary.
5. Evaluate once on the dev-derived test partition.

Leakage safeguards:

- validation labels are not used for validation fitting;
- test labels are never used for selection or fitting;
- row histories are used as supplied and never augmented with same-row clicked candidates;
- prediction outputs are checked for one row per original candidate per baseline;
- candidate order and labels are preserved.

See `docs/offline_evaluation_protocol.md` for details on chronological validation,
impression-grouped metrics, exposure bias, and why logged-candidate evaluation differs from
full-catalog retrieval.

Learning-To-Rank
----------------

Milestone 5 treats every logged candidate as a pointwise binary classification row and ranks
candidates within each impression by predicted click probability. Balanced logistic
regression and histogram gradient boosting combine explainable features from the established
baselines with history coverage and article-text metadata.

Training popularity values are computed chronologically: an impression is featurized before
its click labels update the state. Validation uses train-fitted artifacts only; final test uses
artifacts refit on train plus validation. Configuration selection uses validation NDCG@10 and
MRR, never internal-test metrics.

See [`docs/learning_to_rank_protocol.md`](docs/learning_to_rank_protocol.md) for feature
definitions, leakage controls, commands, and interpretation limits.

Model Selection And Serving
---------------------------

Promotion uses validation NDCG@10, then validation MRR and AUC. A learned policy must improve
validation NDCG@10 by at least 3% relative to the strongest baseline by default. Internal-test
metrics are shown only after selection and never choose the served policy.

The current HGB ranker improves validation NDCG@10 by about 2.77% over category affinity,
which is below the default threshold. Category affinity is therefore packaged as the local
serving policy. Its slightly stronger internal-test NDCG@10 is retained as post-selection
evidence, while HGB's Recall@10 and Hit Rate@10 improvements are also reported.

FastAPI exposes `GET /health`, `GET /policy`, and `POST /rank`. The rank endpoint accepts
history and candidate news IDs, omits unknown candidates explicitly, reports unknown history
IDs, and uses original candidate position for deterministic ties. Startup validates the
manifest and news-catalog schema. Learned-ranker serving is intentionally deferred because
Milestone 5 did not package its full fitted preprocessing state.

See [`docs/model_selection_and_serving.md`](docs/model_selection_and_serving.md) for the
promotion rule, manifest contract, endpoint examples, and limitations.

Monitoring And Observability
----------------------------

Set `FEED_RANKING_OPS_REQUEST_LOG` to enable one aggregate JSONL event per rank request.
Events contain counts, latency, warnings, outcomes, and top-result category aggregates, but
not raw history or candidate IDs. Logging remains optional and a write failure does not fail
ranking.

`GET /metrics` reports process-local request counts, latency summaries, missing-candidate
rate, unknown-history rate, empty-history rate, and log-write errors. Counters reset on
process restart.

The offline monitoring command evaluates the exact packaged policy over validation and
internal test, including slices by history length, candidate count, empty history, and
top-ranked category. Training distributions provide a reference for Jensen-Shannon
divergence and absolute rate-difference checks. These are offline diagnostics, not continuous
production drift monitoring.

See [`docs/monitoring_and_observability.md`](docs/monitoring_and_observability.md) for log
fields, privacy caveats, endpoints, slice definitions, drift checks, and commands.

Full-Catalog Retrieval
----------------------

Milestone 3 starts from the user history and retrieves articles from an eligible catalog.
The clicked candidates in the logged impression are retrieval targets, while the full
original impression candidate list is used only for diagnostics.

Because MIND does not provide reliable publication time in the processed schema, the default
catalog uses observed availability: the first behavior timestamp where an article appears as
an impression candidate. For a query, articles first observed after the query timestamp are
excluded. Equal timestamps are allowed, so current-impression candidates can be eligible.

Article representation uses sparse TF-IDF over title, title plus abstract, or title plus
abstract plus category tokens. Validation vocabulary is fit on training-observed articles
only. Final test vocabulary is refit on train plus validation references after validation
selection. Evaluation labels are never used for fitting.

User profiles average known history article vectors or apply recency-weighted positional
decay. Unknown history IDs are skipped and counted. Empty histories, unknown-only histories,
all-zero profiles, and empty eligible catalogs use popularity fallback based on allowed
fitting-partition positive clicks.

Retrieval metrics:

- Recall@10, Recall@20, Recall@50, Recall@100
- Hit Rate@10, Hit Rate@20, Hit Rate@50, Hit Rate@100
- MRR@100
- NDCG@10, NDCG@20, NDCG@100

Reports include skipped query counts, clicked-target availability, catalog sizes, fallback
usage, history coverage, unique recommendation coverage, exact scoring latency, and sparse
matrix memory diagnostics. TF-IDF indexes and leakage-aware per-query eligibility are reused
across validation configurations, and deterministic partial top-K selection avoids sorting
the complete catalog. Protocol JSON includes end-to-end stage timings. See
`docs/full_catalog_retrieval_protocol.md`.

Current Limitations
-------------------

- The source timestamp timezone is not provided by MIND; parsed timestamps are normalized
  consistently to UTC.
- Entity columns are preserved as raw JSON strings and are not featurized yet.
- Candidate/news coverage gaps are measured and reported, not repaired.
- Logged-candidate ranking does not measure full-catalog retrieval quality.
- Exact full-catalog retrieval is a correctness reference and may not scale without ANN.
- Offline clicks are implicit feedback and carry exposure bias.
- No neural ranking, cache, hosted monitoring, or online experimentation components are implemented yet.
