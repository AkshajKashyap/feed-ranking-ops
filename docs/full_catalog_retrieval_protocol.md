# Full-Catalog Exact Retrieval Protocol

Milestone 3 evaluates candidate retrieval, not second-stage ranking. A retrieval query starts
from a user history and attempts to recover clicked articles from an eligible article
catalog. This differs from Milestone 2, which ranked only the candidates already supplied
inside each logged impression.

## Why Retrieval Is Separate From Ranking

Ranking assumes a candidate set already exists. Retrieval asks whether the system can find
reasonable candidates from a larger catalog in the first place. A strong ranker cannot help
if the clicked article is never retrieved.

## Observed Article Availability

MIND news metadata does not include a trustworthy publication timestamp, so the project does
not invent one. Instead it derives observed availability:

- first candidate timestamp: earliest behavior timestamp where the article appears as an
  impression candidate;
- first history timestamp: earliest behavior timestamp where the article appears in user
  history, recorded as a diagnostic only.

The default `observed_available` catalog includes articles whose first candidate timestamp
is less than or equal to the query timestamp. Current-impression candidates may be eligible
because they are observed at the current timestamp. Articles first observed later are
excluded.

The optional `static_partition_catalog` uses all articles observed in fitting and evaluation
partitions. It is diagnostic, less realistic, and potentially optimistic.

## Article Text Representation

Articles are represented with sparse scikit-learn TF-IDF vectors. Supported text
configurations are:

- `title`
- `title_abstract`
- `title_abstract_category`

The default protocol is inductive: validation TF-IDF vocabulary is fit using articles
referenced by the training partition, and final test vocabulary is refit using train plus
validation references. Evaluation article text may be transformed with the fitted vocabulary,
but evaluation labels are never used for fitting.

## User Profiles

The query vector is built from the row-supplied ordered history only. Current clicked
candidates are not appended to history.

Supported profiles:

- mean-history profile;
- recency-weighted profile with deterministic positional decay.

When history is truncated, the most recent history items are used. Unknown history IDs are
skipped and counted. Empty histories, unknown-only histories, and all-zero vectors fall back
to popularity.

## Exact Retrieval

For each query:

1. read the precomputed eligible catalog boundary;
2. optionally remove history articles;
3. build the user vector;
4. score all eligible article vectors exactly with cosine similarity;
5. return top-K article IDs.

Tie-breaking is deterministic:

1. higher similarity score;
2. earlier observed candidate availability;
3. lexicographically smaller article ID.

Exact retrieval is a correctness reference before approximate nearest-neighbor search. It
may not scale to a large catalog.

## Exact Retrieval Performance

The implementation keeps the protocol exact while avoiding repeated work:

- each TF-IDF article matrix is fit once per text configuration and fitting partition;
- article vectors remain normalized and are reused across history-profile configurations;
- articles are indexed once by first observed candidate timestamp;
- each query's leakage-aware eligible article IDs and matrix row IDs are prepared once and
  reused throughout the validation configuration sweep;
- exact cosine scores are still computed for every eligible article, but deterministic
  partial top-K selection avoids fully sorting the catalog.

The validation and test JSON reports include vectorization, query preparation, scoring,
metric evaluation, article count, query count, eligible-catalog size, and total runtime
timings. These timings make expensive runs diagnosable without changing metrics or selection.

## Popularity Fallback

Fallback ranks eligible articles by positive click counts from the allowed fitting partition.
Validation fallback uses train labels only. Final test fallback uses train plus validation
labels only. Test labels are never used for fitting.

## Metrics And Limitations

Metrics include Recall@10/20/50/100, Hit Rate@10/20/50/100, MRR@100, and NDCG@10/20/100.
They support multiple clicked targets. Queries without valid available clicked targets are
skipped and reported rather than silently contributing zero.

The report also includes catalog sizes, target availability, fallback usage, history
coverage, unique recommendation coverage, and exact-retrieval latency diagnostics.

Offline retrieval recall does not prove online engagement improvement. It is affected by
logged exposure bias and by the fact that clicked targets come from an existing system's
shown candidates.
