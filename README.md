FeedRank Ops
============

A two-stage personalized news-feed recommendation system covering temporal data
preparation, candidate retrieval, ranking, serving, feedback simulation, and monitoring.

Current Scope
-------------

Milestone 1 builds the data foundation for Microsoft MIND-small:

- Parse and validate `news.tsv` and `behaviors.tsv`.
- Audit source data quality and coverage.
- Prepare deterministic chronological train, validation, and offline test splits.
- Write inspectable Parquet outputs and split metadata.

Milestone 2 adds offline logged-candidate ranking evaluation:

- Load and validate processed Parquet outputs.
- Explode nested behavior rows into inspectable candidate-level rows.
- Evaluate MRR, NDCG, recall, hit rate, and impression-level AUC.
- Compare original-order, popularity, time-decayed popularity, category-affinity, and
  TF-IDF content-similarity baselines.
- Tune time-decay half-life on validation only, then refit and evaluate once on test.

This scope intentionally does not implement FAISS, approximate nearest-neighbor retrieval,
two-tower neural models, LightGBM, APIs, Redis, streaming, Docker, dashboards, or monitoring.

Expected Dataset Layout
-----------------------

Download and extract MIND-small manually, then place the files under `data/raw`:

```text
data/raw/
  MINDsmall_train/
    news.tsv
    behaviors.tsv
  MINDsmall_dev/
    news.tsv
    behaviors.tsv
```

The project does not download MIND during tests or normal commands.

Commands
--------

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Validate the expected local file layout:

```bash
python -m feed_ranking_ops.data.validate_layout --data-dir data/raw
make validate-data
```

Audit the dataset:

```bash
python -m feed_ranking_ops.data.audit_dataset --data-dir data/raw --reports-dir reports
make audit-data
```

Prepare chronological splits:

```bash
python -m feed_ranking_ops.data.prepare_dataset \
  --data-dir data/raw \
  --output-dir data/processed \
  --reports-dir reports
make prepare-data
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
recommendation systems. This project therefore:

- treats official MIND train behaviors as model-development data;
- sorts official train behaviors by parsed timestamp;
- assigns the first 80% to train and the remaining 20% to validation by default;
- keeps official MIND dev behaviors as the offline test partition;
- never splits an individual impression row across partitions;
- records timestamp ranges and integrity checks in `split_metadata.json`.

The pipeline reports the observed official dev timestamp range honestly. It does not assume
the official dev period occurs after every official train event unless the local data verifies it.

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

Current Limitations
-------------------

- The source timestamp timezone is not provided by MIND; parsed timestamps are normalized
  consistently to UTC.
- Entity columns are preserved as raw JSON strings and are not featurized yet.
- Candidate/news coverage gaps are measured and reported, not repaired.
- Logged-candidate ranking does not measure full-catalog retrieval quality.
- Offline clicks are implicit feedback and carry exposure bias.
- No retrieval, embedding, API, cache, or monitoring components are implemented yet.
