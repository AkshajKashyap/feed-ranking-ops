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

This milestone intentionally does not implement recommendation models, embeddings,
ranking APIs, Redis, FAISS, or monitoring.

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

Current Limitations
-------------------

- The source timestamp timezone is not provided by MIND; parsed timestamps are normalized
  consistently to UTC.
- Entity columns are preserved as raw JSON strings and are not featurized yet.
- Candidate/news coverage gaps are measured and reported, not repaired.
- No retrieval, ranking, embedding, API, cache, or monitoring components are implemented yet.
