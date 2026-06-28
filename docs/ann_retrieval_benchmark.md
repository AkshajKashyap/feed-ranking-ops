# Dense Vector and FAISS ANN Retrieval Benchmark

Milestone 4 benchmarks approximate nearest-neighbor retrieval without changing the
retrieval task. A query still starts from a user history and tries to recover clicked
articles from a leakage-aware eligible catalog.

## Comparison Chain

The benchmark keeps three layers separate:

1. Sparse TF-IDF exact retrieval from Milestone 3.
2. Dense exact retrieval using the same text/profile rules after TruncatedSVD projection.
3. FAISS retrieval using the identical dense article and query vectors.

Sparse exact to dense exact measures representation loss. Dense exact to FAISS measures
ANN search loss. These are different failure modes and are reported separately.

## Dense Representation

The article TF-IDF index is fit with the Milestone 3 protocol:

- validation vocabulary and SVD fitting use train only;
- final test vocabulary and SVD fitting use train plus validation;
- test labels are never used for configuration selection.

Sparse article vectors are projected with deterministic `TruncatedSVD`. Requested
dimensions such as 32, 64, 128, and 256 are capped to the effective rank supported by the
fitting matrix. Dense article vectors are L2-normalized after projection, and all-zero
vectors are counted in metadata.

Dense user profiles use the same mean-history and recency-history rules as sparse exact
retrieval. Empty histories, unknown histories, and all-zero profiles fall back to the same
popularity fallback used in Milestone 3.

## Dense Exact Reference

Dense exact retrieval uses NumPy matrix multiplication over the eligible catalog. It does
not call FAISS. Tie-breaking is deterministic:

1. higher inner-product score;
2. earlier observed candidate availability;
3. lexicographically smaller article ID.

This is the correctness reference for FAISS Flat and HNSW retrieval.

## FAISS Retrieval

The optional `ann` dependency group installs `faiss-cpu`. Core package imports work without
FAISS. The ANN CLI fails clearly if FAISS is missing.

Implemented index types:

- `IndexFlatIP` as an exact FAISS inner-product baseline;
- `IndexHNSWFlat` with inner-product metric for approximate retrieval.

HNSW exposes `M`, `efConstruction`, and `efSearch`. Retrieval oversamples raw FAISS
neighbors, post-filters by availability and history exclusion, and increases search depth
until top-K is filled or the index is exhausted. Diagnostics count rejected future articles,
history articles, invalid rows, duplicates, raw search calls, and queries unable to fill
top-K.

## Validation Selection

Validation chooses SVD dimension and HNSW parameters only from validation metrics. The
primary selection metric is ANN set recall against dense exact top-K, using top-100 when the
benchmark top-K is at least 100. Clicked-target Recall@K, p95 latency, and index memory are
secondary tie-breakers.

The final test run refits dense representations on train plus validation and evaluates the
selected configuration once on test.

## Fast ANN-Only Smoke Mode

The full comparison deliberately computes sparse exact, dense exact, FAISS Flat, and FAISS
HNSW outputs. That is useful for representation and approximation studies, but unnecessarily
expensive when the goal is to verify one real-data FAISS configuration.

`--ann-only --single-config` runs one explicit dense representation through one FAISS backend:

- validation representation is fit on train and evaluated on validation;
- the final representation is refit on train plus validation and evaluated on the internal test;
- sparse exact and dense exact reference retrieval are skipped;
- no ANN approximation recall or exact-agreement metric is claimed;
- clicked-target retrieval metrics, retrieval rows, diagnostics, and leakage-aware availability
  remain available.

ANN-only defaults favor a quick smoke run:

- `--backend flat`
- `--text-config title`
- `--profile-method mean`
- `--history-mode full`

Use `--backend hnsw` to smoke-test HNSW instead. The full benchmark remains the default when
`--ann-only` is absent.

Stage timing in `protocol.json` covers processed-data loading, availability construction,
TF-IDF vectorization, SVD fitting/projection, FAISS index construction, FAISS search, metrics,
total runtime, query/article counts, average eligible catalog sizes, and peak process memory.

ANN-only search constructs all dense user profiles for a partition once, stacks them into a
query matrix, and calls FAISS in configurable batches. `--query-batch-size` defaults to 256.
This avoids the large thread-dispatch overhead from issuing one small FAISS call per query.

Availability and history exclusion remain query-specific. FAISS overfetches according to
`--oversampling`; queries that still lack top-K eligible results are searched again at a
larger depth in batches. Protocol timing separates profile construction, query-matrix
construction, raw FAISS search, availability filtering, history exclusion, and final top-K
work. It also records mean raw candidates requested and queries returning fewer than top-K.

Coarse progress messages identify data loading, representation fitting, index building,
validation/test search batches, evaluation, and report writing.

## Outputs

The benchmark writes JSON summaries, CSV sweeps, Parquet retrievals, query diagnostics,
runtime metadata, selected configuration metadata, and a Markdown report under the selected
reports directory.

Index metadata records ordered article IDs, dimension, normalization policy, index type and
parameters, fitting provenance, representation metadata, version information, and mapping
fingerprints. Loading validates dimension and article mapping compatibility.

## Commands

```bash
python -m feed_ranking_ops.retrieval.run_ann_benchmark \
  --processed-dir data/processed \
  --reports-dir reports/ann
```

Smoke run:

```bash
python -m feed_ranking_ops.retrieval.run_ann_benchmark \
  --processed-dir data/processed \
  --reports-dir reports/ann_smoke \
  --limit-queries 100 \
  --svd-dims 32,64 \
  --ef-search 64 \
  --oversampling 4
```

Fast real-data ANN-only smoke:

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

This writes `validation_metrics.json`, `test_metrics.json`, `protocol.json`, and
`model_comparison.md` alongside the existing retrieval, diagnostic, latency, index metadata,
and configuration files.

Use:

```bash
python -m pip install -e ".[dev,ann]"
```

before running FAISS benchmarks.
