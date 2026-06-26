# MIND-small Data Split Protocols

FeedRank Ops supports two explicit source and split protocols. Commands never infer or switch
protocols based on the files available locally.

## Official Train/Dev

`official_train_dev` is the default and requires:

- `MINDsmall_train/news.tsv`
- `MINDsmall_train/behaviors.tsv`
- `MINDsmall_dev/news.tsv`
- `MINDsmall_dev/behaviors.tsv`

Official train behavior rows are sorted chronologically. The first 80% are model train and the
remaining 20% are validation. Official dev rows are written to the final offline test partition.

```bash
python -m feed_ranking_ops.data.prepare_dataset \
  --data-dir data/raw \
  --output-dir data/processed \
  --reports-dir reports \
  --protocol official_train_dev
```

## Train-Only Chronological

`train_only_chronological` exists for distributions where the official dev archive is not
available. It requires only:

- `MINDsmall_train/news.tsv`
- `MINDsmall_train/behaviors.tsv`

After chronological sorting, cumulative row boundaries assign the first 70% to train, the next
15% to validation, and the final 15% to an internal holdout. The holdout is stored in
`test_behaviors.parquet` for downstream schema compatibility.

```bash
python -m feed_ranking_ops.data.prepare_dataset \
  --data-dir data/raw \
  --output-dir data/processed \
  --reports-dir reports \
  --protocol train_only_chronological
```

The internal holdout is not the official MIND validation or test benchmark. Metrics from it are
not directly comparable to official MIND validation results.

## Deterministic Boundaries

Behavior rows are ordered by:

1. parsed UTC timestamp;
2. original one-based source-row number;
3. impression ID.

Partition boundaries are row-count boundaries. If several rows share the timestamp at a
boundary, lower source-row numbers remain in the earlier partition. No random sampling is used.
The preparation checks impression IDs and source rows for overlap, verifies chronological
boundaries, and confirms that history ordering and labels remain preserved.

## Outputs

Both protocols write the same files:

- `data/processed/news.parquet`
- `data/processed/train_behaviors.parquet`
- `data/processed/validation_behaviors.parquet`
- `data/processed/test_behaviors.parquet`
- `data/processed/split_metadata.json`
- `reports/split_summary.md`

The metadata records the selected protocol, source splits, requested and observed ratios,
timestamp boundaries, final partition type, reference coverage, and integrity checks.
