from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np

from feed_ranking_ops.retrieval.availability import ArticleAvailability
from feed_ranking_ops.retrieval.dense import DenseArticleIndex

FaissIndexType = Literal["flat", "hnsw"]


class EligibilityLookup(Protocol):
    def __contains__(self, news_id: object) -> bool: ...

    def __len__(self) -> int: ...


class FaissUnavailableError(RuntimeError):
    """Raised when FAISS-backed functionality is requested without faiss-cpu."""


@dataclass(frozen=True)
class FaissIndexConfig:
    index_type: FaissIndexType
    hnsw_m: int = 16
    ef_construction: int = 80
    ef_search: int = 64
    oversampling_factor: int = 4

    def validate(self) -> None:
        if self.index_type not in {"flat", "hnsw"}:
            raise ValueError("index_type must be flat or hnsw")
        if self.hnsw_m <= 0:
            raise ValueError("hnsw_m must be positive")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be positive")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be positive")
        if self.oversampling_factor <= 0:
            raise ValueError("oversampling_factor must be positive")


@dataclass
class FaissSearchResult:
    news_ids: list[str]
    scores: dict[str, float]
    latency_seconds: float
    raw_search_calls: int
    raw_candidates_requested: int
    raw_candidates_examined: int
    rejected_history_count: int
    rejected_unavailable_count: int
    rejected_invalid_count: int
    rejected_duplicate_count: int
    unable_to_fill_top_k: bool
    oversampled_search: bool


@dataclass
class FaissBatchSearchResult:
    results: list[FaissSearchResult]
    batch_count: int
    raw_search_seconds: float
    availability_filter_seconds: float
    history_exclusion_seconds: float
    final_top_k_seconds: float
    other_filter_seconds: float


@dataclass
class LoadedFaissIndex:
    index: Any
    article_ids: list[str]
    dimension: int
    config: FaissIndexConfig
    metadata: dict[str, Any]

    @property
    def article_fingerprint(self) -> str:
        return _fingerprint_article_ids(self.article_ids)

    @property
    def memory_bytes(self) -> int:
        return int(len(self.article_ids) * self.dimension * np.dtype(np.float32).itemsize)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        eligible_news_ids: EligibilityLookup,
        history_news_ids: set[str] | frozenset[str],
        availability: ArticleAvailability,
        top_k: int,
        exclude_history: bool,
    ) -> FaissSearchResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query vector dimension {query.shape[1]} does not match index dimension {self.dimension}"
            )
        start = perf_counter()
        depth = min(self.index.ntotal, max(top_k * self.config.oversampling_factor, top_k))
        depth = max(depth, 1) if self.index.ntotal else 0
        raw_search_calls = 0
        final_scores: dict[int, float] = {}
        final_indices: list[int] = []
        while depth > 0:
            raw_search_calls += 1
            scores, indices = self.index.search(query, depth)
            final_scores = {
                int(index): float(score)
                for score, index in zip(scores[0].tolist(), indices[0].tolist(), strict=True)
                if int(index) >= 0
            }
            final_indices = [int(index) for index in indices[0].tolist() if int(index) >= 0]
            valid = _post_filter_indices(
                final_indices,
                article_ids=self.article_ids,
                scores=final_scores,
                eligible_news_ids=eligible_news_ids,
                history_news_ids=history_news_ids,
                availability=availability,
                exclude_history=exclude_history,
            )
            if len(valid.news_ids) >= top_k or depth >= self.index.ntotal:
                break
            depth = min(self.index.ntotal, max(depth + 1, depth * 2))
        filtered = _post_filter_indices(
            final_indices,
            article_ids=self.article_ids,
            scores=final_scores,
            eligible_news_ids=eligible_news_ids,
            history_news_ids=history_news_ids,
            availability=availability,
            exclude_history=exclude_history,
        )
        news_ids = filtered.news_ids[:top_k]
        return FaissSearchResult(
            news_ids=news_ids,
            scores={news_id: filtered.scores[news_id] for news_id in news_ids},
            latency_seconds=perf_counter() - start,
            raw_search_calls=raw_search_calls,
            raw_candidates_requested=depth,
            raw_candidates_examined=len(final_indices),
            rejected_history_count=filtered.rejected_history_count,
            rejected_unavailable_count=filtered.rejected_unavailable_count,
            rejected_invalid_count=filtered.rejected_invalid_count,
            rejected_duplicate_count=filtered.rejected_duplicate_count,
            unable_to_fill_top_k=len(news_ids) < top_k and len(news_ids) < len(eligible_news_ids),
            oversampled_search=depth > top_k,
        )

    def search_batch(
        self,
        query_vectors: np.ndarray,
        *,
        eligible_news_ids: list[EligibilityLookup],
        history_news_ids: list[set[str] | frozenset[str]],
        availability: ArticleAvailability,
        top_k: int,
        exclude_history: bool,
        batch_size: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> FaissBatchSearchResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError(
                "query vector matrix must be 2D with the FAISS index dimension"
            )
        if not (
            len(queries) == len(eligible_news_ids) == len(history_news_ids)
        ):
            raise ValueError("query vectors and filtering inputs must have equal lengths")

        results: list[FaissSearchResult] = []
        raw_search_seconds = 0.0
        availability_filter_seconds = 0.0
        history_exclusion_seconds = 0.0
        final_top_k_seconds = 0.0
        other_filter_seconds = 0.0
        batch_count = 0
        total = len(queries)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            chunk = self._search_batch_chunk(
                queries[start:stop],
                eligible_news_ids=eligible_news_ids[start:stop],
                history_news_ids=history_news_ids[start:stop],
                availability=availability,
                top_k=top_k,
                exclude_history=exclude_history,
            )
            results.extend(chunk.results)
            raw_search_seconds += chunk.raw_search_seconds
            availability_filter_seconds += chunk.availability_filter_seconds
            history_exclusion_seconds += chunk.history_exclusion_seconds
            final_top_k_seconds += chunk.final_top_k_seconds
            other_filter_seconds += chunk.other_filter_seconds
            batch_count += 1
            if progress_callback is not None:
                progress_callback(stop, total)
        return FaissBatchSearchResult(
            results=results,
            batch_count=batch_count,
            raw_search_seconds=raw_search_seconds,
            availability_filter_seconds=availability_filter_seconds,
            history_exclusion_seconds=history_exclusion_seconds,
            final_top_k_seconds=final_top_k_seconds,
            other_filter_seconds=other_filter_seconds,
        )

    def _search_batch_chunk(
        self,
        query_vectors: np.ndarray,
        *,
        eligible_news_ids: list[EligibilityLookup],
        history_news_ids: list[set[str] | frozenset[str]],
        availability: ArticleAvailability,
        top_k: int,
        exclude_history: bool,
    ) -> FaissBatchSearchResult:
        count = len(query_vectors)
        if count == 0:
            return FaissBatchSearchResult([], 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        depth = min(
            self.index.ntotal,
            max(top_k * self.config.oversampling_factor, top_k),
        )
        depth = max(depth, 1) if self.index.ntotal else 0
        pending = np.arange(count, dtype=np.int64)
        filtered_by_query: list[_FilteredCandidates | None] = [None] * count
        raw_search_calls = np.zeros(count, dtype=np.int32)
        raw_candidates_requested = np.zeros(count, dtype=np.int32)
        raw_candidates_examined = np.zeros(count, dtype=np.int32)
        per_query_latency = np.zeros(count, dtype=np.float64)
        raw_search_seconds = 0.0
        availability_filter_seconds = 0.0
        history_exclusion_seconds = 0.0
        final_top_k_seconds = 0.0
        other_filter_seconds = 0.0

        while depth > 0 and len(pending):
            search_start = perf_counter()
            scores, indices = self.index.search(query_vectors[pending], depth)
            search_elapsed = perf_counter() - search_start
            raw_search_seconds += search_elapsed
            per_query_latency[pending] += search_elapsed / len(pending)
            next_pending: list[int] = []
            for local_index, query_index_value in enumerate(pending):
                query_index = int(query_index_value)
                filter_start = perf_counter()
                row_indices = [
                    int(index) for index in indices[local_index].tolist() if int(index) >= 0
                ]
                score_by_row = {
                    int(index): float(score)
                    for score, index in zip(
                        scores[local_index].tolist(),
                        indices[local_index].tolist(),
                        strict=True,
                    )
                    if int(index) >= 0
                }
                filtered, filter_timing = _post_filter_indices_profiled(
                    row_indices,
                    article_ids=self.article_ids,
                    scores=score_by_row,
                    eligible_news_ids=eligible_news_ids[query_index],
                    history_news_ids=history_news_ids[query_index],
                    availability=availability,
                    exclude_history=exclude_history,
                )
                filter_elapsed = perf_counter() - filter_start
                per_query_latency[query_index] += filter_elapsed
                filtered_by_query[query_index] = filtered
                raw_search_calls[query_index] += 1
                raw_candidates_requested[query_index] = depth
                raw_candidates_examined[query_index] = len(row_indices)
                availability_filter_seconds += filter_timing[
                    "availability_filter_seconds"
                ]
                history_exclusion_seconds += filter_timing[
                    "history_exclusion_seconds"
                ]
                final_top_k_seconds += filter_timing["final_top_k_seconds"]
                other_filter_seconds += max(
                    0.0,
                    filter_elapsed
                    - filter_timing["availability_filter_seconds"]
                    - filter_timing["history_exclusion_seconds"]
                    - filter_timing["final_top_k_seconds"],
                )
                if (
                    len(filtered.news_ids) < top_k
                    and depth < self.index.ntotal
                    and len(filtered.news_ids)
                    < len(eligible_news_ids[query_index])
                ):
                    next_pending.append(query_index)
            pending = np.asarray(next_pending, dtype=np.int64)
            if len(pending):
                depth = min(self.index.ntotal, max(depth + 1, depth * 2))

        results: list[FaissSearchResult] = []
        for query_index, filtered in enumerate(filtered_by_query):
            if filtered is None:
                filtered = _FilteredCandidates([], {}, 0, 0, 0, 0)
            truncation_start = perf_counter()
            news_ids = filtered.news_ids[:top_k]
            score_by_id = {
                news_id: filtered.scores[news_id] for news_id in news_ids
            }
            truncation_elapsed = perf_counter() - truncation_start
            final_top_k_seconds += truncation_elapsed
            per_query_latency[query_index] += truncation_elapsed
            results.append(
                FaissSearchResult(
                    news_ids=news_ids,
                    scores=score_by_id,
                    latency_seconds=float(per_query_latency[query_index]),
                    raw_search_calls=int(raw_search_calls[query_index]),
                    raw_candidates_requested=int(raw_candidates_requested[query_index]),
                    raw_candidates_examined=int(raw_candidates_examined[query_index]),
                    rejected_history_count=filtered.rejected_history_count,
                    rejected_unavailable_count=filtered.rejected_unavailable_count,
                    rejected_invalid_count=filtered.rejected_invalid_count,
                    rejected_duplicate_count=filtered.rejected_duplicate_count,
                    unable_to_fill_top_k=(
                        len(news_ids) < top_k
                        and len(news_ids) < len(eligible_news_ids[query_index])
                    ),
                    oversampled_search=raw_candidates_requested[query_index] > top_k,
                )
            )
        return FaissBatchSearchResult(
            results=results,
            batch_count=1,
            raw_search_seconds=raw_search_seconds,
            availability_filter_seconds=availability_filter_seconds,
            history_exclusion_seconds=history_exclusion_seconds,
            final_top_k_seconds=final_top_k_seconds,
            other_filter_seconds=other_filter_seconds,
        )


@dataclass
class _FilteredCandidates:
    news_ids: list[str]
    scores: dict[str, float]
    rejected_history_count: int
    rejected_unavailable_count: int
    rejected_invalid_count: int
    rejected_duplicate_count: int


def require_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FaissUnavailableError(
            "FAISS is required for ANN retrieval. Install it with: "
            'python -m pip install -e ".[ann]"'
        ) from exc
    return faiss


def set_faiss_threads(num_threads: int | None) -> None:
    if num_threads is None:
        return
    if num_threads <= 0:
        raise ValueError("faiss_threads must be positive when provided")
    faiss = require_faiss()
    faiss.omp_set_num_threads(num_threads)


def build_faiss_index(
    dense_index: DenseArticleIndex,
    config: FaissIndexConfig,
) -> LoadedFaissIndex:
    config.validate()
    faiss = require_faiss()
    dimension = dense_index.effective_dimension
    if config.index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexHNSWFlat(dimension, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.ef_construction
        index.hnsw.efSearch = config.ef_search
    vectors = np.ascontiguousarray(dense_index.vectors, dtype=np.float32)
    if vectors.shape[1] != dimension:
        raise ValueError("dense vector matrix dimension does not match metadata")
    index.add(vectors)
    metadata = make_index_metadata(dense_index, config)
    return LoadedFaissIndex(
        index=index,
        article_ids=list(dense_index.article_ids),
        dimension=dimension,
        config=config,
        metadata=metadata,
    )


def save_faiss_index(
    loaded_index: LoadedFaissIndex,
    *,
    index_path: Path,
    metadata_path: Path,
) -> None:
    faiss = require_faiss()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(loaded_index.index, str(index_path))
    metadata = dict(loaded_index.metadata)
    metadata["index_path"] = str(index_path.name)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_faiss_index(
    *,
    index_path: Path,
    metadata_path: Path,
    expected_dimension: int | None = None,
    expected_article_fingerprint: str | None = None,
) -> LoadedFaissIndex:
    faiss = require_faiss()
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing FAISS metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dimension = int(metadata["dimension"])
    article_ids = list(metadata["article_ids"])
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"FAISS index dimension {dimension} does not match expected {expected_dimension}"
        )
    if (
        expected_article_fingerprint is not None
        and metadata.get("article_fingerprint") != expected_article_fingerprint
    ):
        raise ValueError("FAISS article mapping fingerprint does not match expected mapping")
    config = FaissIndexConfig(**metadata["index_config"])
    index = faiss.read_index(str(index_path))
    if index.d != dimension:
        raise ValueError(f"FAISS index file dimension {index.d} does not match metadata {dimension}")
    if index.ntotal != len(article_ids):
        raise ValueError(
            f"FAISS index row count {index.ntotal} does not match metadata article count {len(article_ids)}"
        )
    return LoadedFaissIndex(
        index=index,
        article_ids=article_ids,
        dimension=dimension,
        config=config,
        metadata=metadata,
    )


def make_index_metadata(
    dense_index: DenseArticleIndex,
    config: FaissIndexConfig,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "index_type": config.index_type,
        "index_config": asdict(config),
        "dimension": dense_index.effective_dimension,
        "normalization": "l2_row_normalized",
        "article_ids": list(dense_index.article_ids),
        "article_fingerprint": dense_index.article_fingerprint,
        "representation": dense_index.metadata(),
        "metadata_fingerprint": _metadata_fingerprint(
            dense_index.article_ids,
            dense_index.effective_dimension,
            asdict(config),
        ),
        "versions": _versions(),
    }


def _post_filter_indices(
    indices: list[int],
    *,
    article_ids: list[str],
    scores: dict[int, float],
    eligible_news_ids: EligibilityLookup,
    history_news_ids: set[str] | frozenset[str],
    availability: ArticleAvailability,
    exclude_history: bool,
) -> _FilteredCandidates:
    filtered, _timing = _post_filter_indices_profiled(
        indices,
        article_ids=article_ids,
        scores=scores,
        eligible_news_ids=eligible_news_ids,
        history_news_ids=history_news_ids,
        availability=availability,
        exclude_history=exclude_history,
    )
    return filtered


def _post_filter_indices_profiled(
    indices: list[int],
    *,
    article_ids: list[str],
    scores: dict[int, float],
    eligible_news_ids: EligibilityLookup,
    history_news_ids: set[str] | frozenset[str],
    availability: ArticleAvailability,
    exclude_history: bool,
) -> tuple[_FilteredCandidates, dict[str, float]]:
    total_start = perf_counter()
    seen: set[str] = set()
    unique_candidates: list[tuple[int, str]] = []
    rejected_invalid = 0
    rejected_duplicate = 0
    for index in indices:
        if index < 0 or index >= len(article_ids):
            rejected_invalid += 1
            continue
        news_id = article_ids[index]
        if news_id in seen:
            rejected_duplicate += 1
            continue
        seen.add(news_id)
        unique_candidates.append((index, news_id))

    availability_start = perf_counter()
    available_candidates = [
        (index, news_id)
        for index, news_id in unique_candidates
        if news_id in eligible_news_ids
    ]
    availability_filter_seconds = perf_counter() - availability_start
    rejected_unavailable = len(unique_candidates) - len(available_candidates)

    history_start = perf_counter()
    if exclude_history:
        retained_candidates = [
            (index, news_id)
            for index, news_id in available_candidates
            if news_id not in history_news_ids
        ]
    else:
        retained_candidates = available_candidates
    history_exclusion_seconds = perf_counter() - history_start
    rejected_history = len(available_candidates) - len(retained_candidates)

    ranking_start = perf_counter()
    score_by_id = {
        news_id: scores.get(index, 0.0) for index, news_id in retained_candidates
    }
    ranked = sorted(
        score_by_id,
        key=lambda news_id: (
            -score_by_id[news_id],
            availability.first_candidate_timestamp.get(news_id, datetime.max),
            news_id,
        ),
    )
    final_top_k_seconds = perf_counter() - ranking_start
    total_seconds = perf_counter() - total_start
    measured_seconds = (
        availability_filter_seconds
        + history_exclusion_seconds
        + final_top_k_seconds
    )
    return _FilteredCandidates(
        news_ids=ranked,
        scores=score_by_id,
        rejected_history_count=rejected_history,
        rejected_unavailable_count=rejected_unavailable,
        rejected_invalid_count=rejected_invalid,
        rejected_duplicate_count=rejected_duplicate,
    ), {
        "availability_filter_seconds": availability_filter_seconds,
        "history_exclusion_seconds": history_exclusion_seconds,
        "final_top_k_seconds": final_top_k_seconds,
        "other_filter_seconds": max(0.0, total_seconds - measured_seconds),
    }


def _fingerprint_article_ids(article_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(article_ids).encode("utf-8")).hexdigest()


def _metadata_fingerprint(
    article_ids: list[str],
    dimension: int,
    config: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "article_ids": article_ids,
            "dimension": dimension,
            "config": config,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _versions() -> dict[str, str | None]:
    faiss = require_faiss()
    try:
        import sklearn
    except ImportError:
        sklearn_version = None
    else:
        sklearn_version = sklearn.__version__
    return {
        "faiss": getattr(faiss, "__version__", None),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
    }
