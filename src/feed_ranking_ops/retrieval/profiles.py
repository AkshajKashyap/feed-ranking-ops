from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse

from feed_ranking_ops.retrieval.text import ArticleTextIndex

ProfileType = Literal["mean", "recency"]


@dataclass(frozen=True)
class HistoryProfileConfig:
    profile_type: ProfileType
    max_history_length: int | None = None
    decay: float | None = None
    deduplicate_history: bool = False


@dataclass
class UserProfileResult:
    vector: sparse.csr_matrix | None
    known_history_count: int
    unknown_history_count: int
    fallback_reason: str | None = None


def build_user_profile(
    history_news_ids: list[str],
    index: ArticleTextIndex,
    config: HistoryProfileConfig,
) -> UserProfileResult:
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
        return UserProfileResult(None, 0, 0, "empty_history")
    if not known_rows:
        return UserProfileResult(None, 0, unknown_count, "no_known_history")
    matrix = index.article_matrix[known_rows]
    if matrix.shape[1] == 0 or matrix.nnz == 0:
        return UserProfileResult(None, len(known_rows), unknown_count, "all_zero_profile")
    if config.profile_type == "mean":
        profile = matrix.mean(axis=0)
    elif config.profile_type == "recency":
        if config.decay is None or not 0 < config.decay <= 1:
            raise ValueError("recency profile requires decay in (0, 1]")
        weights = _recency_weights(len(known_rows), config.decay)
        profile = weights @ matrix
    else:
        raise ValueError(f"Unknown profile type: {config.profile_type}")
    profile = sparse.csr_matrix(profile)
    if profile.nnz == 0 or float(np.linalg.norm(profile.data)) == 0.0:
        return UserProfileResult(
            None,
            len(known_rows),
            unknown_count,
            "all_zero_profile",
        )
    return UserProfileResult(profile, len(known_rows), unknown_count)


def _truncate_history(
    history_news_ids: list[str],
    max_history_length: int | None,
) -> list[str]:
    if max_history_length is None:
        return list(history_news_ids)
    if max_history_length <= 0:
        raise ValueError("max_history_length must be positive or None")
    return list(history_news_ids[-max_history_length:])


def _recency_weights(count: int, decay: float) -> sparse.csr_matrix:
    # The last history item is most recent and receives weight 1.
    raw = np.array([decay ** (count - index - 1) for index in range(count)], dtype=float)
    raw = raw / raw.sum()
    return sparse.csr_matrix(raw.reshape(1, count))
