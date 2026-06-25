from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD

from feed_ranking_ops.evaluation.processed import BehaviorImpression, NewsItem
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.text import ArticleTextIndex, TextConfig, fit_article_text_index


@dataclass(frozen=True)
class DenseRepresentationConfig:
    requested_dimension: int
    text_config: TextConfig
    seed: int = 42


@dataclass
class DenseArticleIndex:
    representation_config: DenseRepresentationConfig
    article_ids: list[str]
    article_to_row: dict[str, int]
    vectors: np.ndarray
    zero_vector_article_ids: list[str]
    effective_dimension: int
    fitting_article_ids: list[str]
    fitting_partitions: list[str]
    tfidf_index: ArticleTextIndex
    svd: TruncatedSVD | None

    @property
    def requested_dimension(self) -> int:
        return self.representation_config.requested_dimension

    @property
    def vector_memory_bytes(self) -> int:
        return int(self.vectors.nbytes)

    @property
    def article_fingerprint(self) -> str:
        return fingerprint_article_ids(self.article_ids)

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_dimension": self.requested_dimension,
            "effective_dimension": self.effective_dimension,
            "text_config": asdict(self.representation_config.text_config),
            "seed": self.representation_config.seed,
            "article_count": len(self.article_ids),
            "zero_vector_article_count": len(self.zero_vector_article_ids),
            "fitting_article_count": len(self.fitting_article_ids),
            "fitting_partitions": list(self.fitting_partitions),
            "tfidf_vocabulary_size": self.tfidf_index.vocabulary_size,
            "article_fingerprint": self.article_fingerprint,
            "approx_vector_memory_bytes": self.vector_memory_bytes,
        }


@dataclass
class DenseUserProfileResult:
    vector: np.ndarray | None
    known_history_count: int
    unknown_history_count: int
    fallback_reason: str | None = None


def fit_dense_article_index(
    *,
    news: dict[str, NewsItem],
    fitting_behaviors: list[BehaviorImpression],
    text_config: TextConfig,
    requested_dimension: int,
    seed: int,
    fitting_partitions: list[str],
) -> DenseArticleIndex:
    if requested_dimension <= 0:
        raise ValueError("requested_dimension must be positive")
    tfidf_index = fit_article_text_index(
        news=news,
        fitting_behaviors=fitting_behaviors,
        text_config=text_config,
    )
    dense, svd, effective_dimension = _project_tfidf_to_dense(
        tfidf_index,
        requested_dimension=requested_dimension,
        seed=seed,
    )
    dense, zero_mask = l2_normalize_rows(dense)
    zero_ids = [
        article_id
        for article_id, is_zero in zip(tfidf_index.article_ids, zero_mask, strict=True)
        if is_zero
    ]
    return DenseArticleIndex(
        representation_config=DenseRepresentationConfig(
            requested_dimension=requested_dimension,
            text_config=text_config,
            seed=seed,
        ),
        article_ids=list(tfidf_index.article_ids),
        article_to_row=dict(tfidf_index.article_to_row),
        vectors=dense,
        zero_vector_article_ids=zero_ids,
        effective_dimension=effective_dimension,
        fitting_article_ids=list(tfidf_index.fitting_article_ids),
        fitting_partitions=list(fitting_partitions),
        tfidf_index=tfidf_index,
        svd=svd,
    )


def build_dense_user_profile(
    history_news_ids: list[str],
    index: DenseArticleIndex,
    config: HistoryProfileConfig,
) -> DenseUserProfileResult:
    history = _truncate_history(history_news_ids, config.max_history_length)
    if config.deduplicate_history:
        history = list(dict.fromkeys(history))
    known_rows = [
        index.article_to_row[news_id]
        for news_id in history
        if news_id in index.article_to_row
    ]
    unknown_count = len(history) - len(known_rows)
    if not history:
        return DenseUserProfileResult(None, 0, 0, "empty_history")
    if not known_rows:
        return DenseUserProfileResult(None, 0, unknown_count, "no_known_history")
    matrix = index.vectors[known_rows]
    if matrix.shape[1] == 0 or not np.any(matrix):
        return DenseUserProfileResult(
            None,
            len(known_rows),
            unknown_count,
            "all_zero_profile",
        )
    if config.profile_type == "mean":
        profile = matrix.mean(axis=0)
    elif config.profile_type == "recency":
        if config.decay is None or not 0 < config.decay <= 1:
            raise ValueError("recency profile requires decay in (0, 1]")
        weights = _recency_weights(len(known_rows), config.decay)
        profile = weights @ matrix
    else:
        raise ValueError(f"Unknown profile type: {config.profile_type}")
    profile = np.asarray(profile, dtype=np.float32).reshape(1, -1)
    profile, zero_mask = l2_normalize_rows(profile)
    if bool(zero_mask[0]):
        return DenseUserProfileResult(
            None,
            len(known_rows),
            unknown_count,
            "all_zero_profile",
        )
    return DenseUserProfileResult(profile.ravel(), len(known_rows), unknown_count)


def l2_normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a 2D matrix")
    norms = np.linalg.norm(array, axis=1)
    zero_mask = norms <= 1e-12
    normalized = array.copy()
    if np.any(~zero_mask):
        normalized[~zero_mask] = normalized[~zero_mask] / norms[~zero_mask, None]
    return np.ascontiguousarray(normalized, dtype=np.float32), zero_mask


def fingerprint_article_ids(article_ids: list[str]) -> str:
    payload = "\n".join(article_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _project_tfidf_to_dense(
    index: ArticleTextIndex,
    *,
    requested_dimension: int,
    seed: int,
) -> tuple[np.ndarray, TruncatedSVD | None, int]:
    if index.article_matrix.shape[1] == 0:
        return np.zeros((len(index.article_ids), 1), dtype=np.float32), None, 1
    fitting_rows = [
        index.article_to_row[article_id]
        for article_id in index.fitting_article_ids
        if article_id in index.article_to_row
    ]
    if not fitting_rows:
        return np.zeros((len(index.article_ids), 1), dtype=np.float32), None, 1
    fitting_matrix = index.article_matrix[fitting_rows]
    if fitting_matrix.nnz == 0:
        return np.zeros((len(index.article_ids), 1), dtype=np.float32), None, 1
    max_components = max(1, min(fitting_matrix.shape))
    effective_dimension = min(requested_dimension, max_components)
    svd = TruncatedSVD(n_components=effective_dimension, random_state=seed)
    svd.fit(fitting_matrix)
    dense = svd.transform(index.article_matrix)
    return np.asarray(dense, dtype=np.float32), svd, effective_dimension


def _truncate_history(
    history_news_ids: list[str],
    max_history_length: int | None,
) -> list[str]:
    if max_history_length is None:
        return list(history_news_ids)
    if max_history_length <= 0:
        raise ValueError("max_history_length must be positive or None")
    return list(history_news_ids[-max_history_length:])


def _recency_weights(count: int, decay: float) -> np.ndarray:
    raw = np.array([decay ** (count - index - 1) for index in range(count)], dtype=np.float32)
    return raw / raw.sum()
