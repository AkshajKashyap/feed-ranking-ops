from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from feed_ranking_ops.retrieval.availability import ArticleAvailability
from feed_ranking_ops.retrieval.dense import DenseArticleIndex

FaissIndexType = Literal["flat", "hnsw"]


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
    raw_candidates_examined: int
    rejected_history_count: int
    rejected_unavailable_count: int
    rejected_invalid_count: int
    rejected_duplicate_count: int
    unable_to_fill_top_k: bool
    oversampled_search: bool


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
        eligible_news_ids: set[str],
        history_news_ids: set[str],
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
            raw_candidates_examined=len(final_indices),
            rejected_history_count=filtered.rejected_history_count,
            rejected_unavailable_count=filtered.rejected_unavailable_count,
            rejected_invalid_count=filtered.rejected_invalid_count,
            rejected_duplicate_count=filtered.rejected_duplicate_count,
            unable_to_fill_top_k=len(news_ids) < top_k and len(news_ids) < len(eligible_news_ids),
            oversampled_search=depth > top_k,
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
    eligible_news_ids: set[str],
    history_news_ids: set[str],
    availability: ArticleAvailability,
    exclude_history: bool,
) -> _FilteredCandidates:
    seen: set[str] = set()
    score_by_id: dict[str, float] = {}
    rejected_invalid = 0
    rejected_history = 0
    rejected_unavailable = 0
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
        if news_id not in eligible_news_ids:
            rejected_unavailable += 1
            continue
        if exclude_history and news_id in history_news_ids:
            rejected_history += 1
            continue
        score_by_id[news_id] = scores.get(index, 0.0)
    ranked = sorted(
        score_by_id,
        key=lambda news_id: (
            -score_by_id[news_id],
            availability.first_candidate_timestamp.get(news_id, datetime.max),
            news_id,
        ),
    )
    return _FilteredCandidates(
        news_ids=ranked,
        scores=score_by_id,
        rejected_history_count=rejected_history,
        rejected_unavailable_count=rejected_unavailable,
        rejected_invalid_count=rejected_invalid,
        rejected_duplicate_count=rejected_duplicate,
    )


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
