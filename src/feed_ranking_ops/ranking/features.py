from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from feed_ranking_ops.evaluation.baselines import TfidfContentSimilarityBaseline
from feed_ranking_ops.evaluation.processed import BehaviorImpression, NewsItem

FEATURE_NAMES = [
    "candidate_position",
    "reciprocal_position",
    "global_click_count",
    "time_decayed_click_count",
    "category_history_count",
    "subcategory_history_count",
    "tfidf_history_similarity",
    "category_article_frequency",
    "subcategory_article_frequency",
    "history_length",
    "known_history_count",
    "empty_history",
    "candidate_seen_in_history",
    "has_abstract",
    "title_length",
    "abstract_length",
]


@dataclass(frozen=True)
class CandidateFeatureMatrix:
    features: np.ndarray
    labels: np.ndarray
    behaviors: list[BehaviorImpression]
    offsets: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]

    @property
    def n_candidates(self) -> int:
        return int(self.features.shape[0])


class CandidateFeatureBuilder:
    """Build explainable candidate features under an explicit fitting boundary."""

    def __init__(self, *, half_life_hours: float = 24.0) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self.half_life_hours = half_life_hours
        self.global_click_counts: Counter[str] = Counter()
        self.decayed_click_counts: Counter[str] = Counter()
        self.category_article_counts: Counter[str] = Counter()
        self.subcategory_article_counts: Counter[str] = Counter()
        self.tfidf = TfidfContentSimilarityBaseline()
        self.fitting_partitions: list[str] = []
        self.fitted = False
        self.fit_cutoff: datetime | None = None

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> CandidateFeatureBuilder:
        if not behaviors:
            raise ValueError("Feature fitting requires at least one behavior")
        actual_partitions = sorted({behavior.partition for behavior in behaviors})
        if not set(actual_partitions).issubset(fitting_partitions):
            raise ValueError(
                "Feature fitting behaviors include a partition outside fitting_partitions"
            )
        self.fitting_partitions = list(fitting_partitions)
        self.fit_cutoff = max(behavior.timestamp for behavior in behaviors)
        self.global_click_counts = Counter(
            candidate.news_id
            for behavior in behaviors
            for candidate in behavior.candidates
            if candidate.clicked == 1
        )
        self.decayed_click_counts = Counter()
        for behavior in behaviors:
            age_hours = max(
                (self.fit_cutoff - behavior.timestamp).total_seconds() / 3600.0,
                0.0,
            )
            weight = 0.5 ** (age_hours / self.half_life_hours)
            for candidate in behavior.candidates:
                if candidate.clicked == 1:
                    self.decayed_click_counts[candidate.news_id] += weight

        observed_news_ids = {
            news_id
            for behavior in behaviors
            for news_id in [
                *behavior.history_news_ids,
                *(candidate.news_id for candidate in behavior.candidates),
            ]
            if news_id in news
        }
        self.category_article_counts = Counter(
            news[news_id].category for news_id in observed_news_ids
        )
        self.subcategory_article_counts = Counter(
            news[news_id].subcategory for news_id in observed_news_ids
        )
        self.tfidf.fit(
            behaviors,
            news,
            fitting_partitions=fitting_partitions,
        )
        self.fitted = True
        return self

    def transform(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        chronological_training: bool = False,
    ) -> CandidateFeatureMatrix:
        if not self.fitted:
            raise RuntimeError("CandidateFeatureBuilder must be fitted before transform")
        if chronological_training:
            _validate_chronological_order(behaviors)

        n_candidates = sum(len(behavior.candidates) for behavior in behaviors)
        features = np.zeros((n_candidates, len(FEATURE_NAMES)), dtype=np.float32)
        labels = np.full(n_candidates, -1, dtype=np.int8)
        offsets = np.asarray(
            [
                0,
                *np.cumsum(
                    [len(behavior.candidates) for behavior in behaviors],
                    dtype=np.int64,
                ),
            ],
            dtype=np.int64,
        )
        tfidf_scores = self._tfidf_scores(behaviors, offsets)
        prior_clicks: Counter[str] = Counter()
        prior_decayed: dict[str, tuple[float, datetime]] = {}
        row_index = 0

        for behavior_index, behavior in enumerate(behaviors):
            category_counts, subcategory_counts, known_history = _history_counts(
                behavior,
                news,
            )
            history_set = set(behavior.history_news_ids)
            for candidate in behavior.candidates:
                item = news[candidate.news_id]
                if chronological_training:
                    global_click_count = float(prior_clicks[candidate.news_id])
                    decayed_click_count = _prior_decayed_score(
                        prior_decayed,
                        candidate.news_id,
                        behavior.timestamp,
                        self.half_life_hours,
                    )
                else:
                    global_click_count = float(
                        self.global_click_counts[candidate.news_id]
                    )
                    decayed_click_count = float(
                        self.decayed_click_counts[candidate.news_id]
                    )
                features[row_index] = [
                    float(candidate.position),
                    1.0 / (float(candidate.position) + 1.0),
                    global_click_count,
                    decayed_click_count,
                    float(category_counts[item.category]),
                    float(subcategory_counts[item.subcategory]),
                    float(tfidf_scores[row_index]),
                    float(self.category_article_counts[item.category]),
                    float(self.subcategory_article_counts[item.subcategory]),
                    float(len(behavior.history_news_ids)),
                    float(known_history),
                    float(not behavior.history_news_ids),
                    float(candidate.news_id in history_set),
                    float(bool(item.abstract.strip())),
                    float(len(item.title)),
                    float(len(item.abstract)),
                ]
                labels[row_index] = (
                    -1 if candidate.clicked is None else int(candidate.clicked)
                )
                row_index += 1

            if chronological_training:
                _update_prior_click_features(
                    prior_clicks,
                    prior_decayed,
                    behavior,
                    self.half_life_hours,
                )

        metadata = {
            "fitting_partitions": list(self.fitting_partitions),
            "fit_cutoff": self.fit_cutoff.isoformat() if self.fit_cutoff else None,
            "chronological_training_features": chronological_training,
            "current_impression_labels_used": False,
            "future_clicks_used": False,
            "text_policy": self.tfidf.config()["vocabulary_policy"],
            "half_life_hours": self.half_life_hours,
            "n_impressions": len(behaviors),
            "n_candidates": n_candidates,
        }
        return CandidateFeatureMatrix(
            features=features,
            labels=labels,
            behaviors=behaviors,
            offsets=offsets,
            feature_names=list(FEATURE_NAMES),
            metadata=metadata,
        )

    def _tfidf_scores(
        self,
        behaviors: list[BehaviorImpression],
        offsets: np.ndarray,
        *,
        candidate_chunk_size: int = 100_000,
    ) -> np.ndarray:
        n_candidates = int(offsets[-1])
        scores = np.zeros(n_candidates, dtype=np.float32)
        article_matrix = self.tfidf.article_matrix
        if article_matrix is None or not self.tfidf.article_index:
            return scores

        history_rows: list[int] = []
        history_columns: list[int] = []
        candidate_articles = np.full(n_candidates, -1, dtype=np.int32)
        candidate_profiles = np.empty(n_candidates, dtype=np.int32)
        candidate_row = 0
        for behavior_index, behavior in enumerate(behaviors):
            for news_id in behavior.history_news_ids:
                article_index = self.tfidf.article_index.get(news_id)
                if article_index is not None:
                    history_rows.append(behavior_index)
                    history_columns.append(article_index)
            for candidate in behavior.candidates:
                article_index = self.tfidf.article_index.get(candidate.news_id)
                if article_index is not None:
                    candidate_articles[candidate_row] = article_index
                candidate_profiles[candidate_row] = behavior_index
                candidate_row += 1

        history_assignment = csr_matrix(
            (
                np.ones(len(history_rows), dtype=np.float32),
                (history_rows, history_columns),
            ),
            shape=(len(behaviors), article_matrix.shape[0]),
        )
        profiles = normalize(
            history_assignment @ article_matrix,
            norm="l2",
            copy=False,
        ).tocsr()
        for start in range(0, n_candidates, candidate_chunk_size):
            end = min(start + candidate_chunk_size, n_candidates)
            article_indices = candidate_articles[start:end]
            valid = article_indices >= 0
            if not np.any(valid):
                continue
            local_rows = np.flatnonzero(valid)
            similarities = article_matrix[article_indices[valid]].multiply(
                profiles[candidate_profiles[start:end][valid]]
            )
            scores[start + local_rows] = np.asarray(
                similarities.sum(axis=1)
            ).ravel()
        return scores


def _history_counts(
    behavior: BehaviorImpression,
    news: dict[str, NewsItem],
) -> tuple[Counter[str], Counter[str], int]:
    categories: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    known = 0
    for news_id in behavior.history_news_ids:
        item = news.get(news_id)
        if item is None:
            continue
        known += 1
        categories[item.category] += 1
        subcategories[item.subcategory] += 1
    return categories, subcategories, known


def _validate_chronological_order(behaviors: list[BehaviorImpression]) -> None:
    timestamps = [behavior.timestamp for behavior in behaviors]
    if timestamps != sorted(timestamps):
        raise ValueError(
            "chronological_training requires behaviors in nondecreasing timestamp order"
        )


def _prior_decayed_score(
    scores: dict[str, tuple[float, datetime]],
    news_id: str,
    timestamp: datetime,
    half_life_hours: float,
) -> float:
    state = scores.get(news_id)
    if state is None:
        return 0.0
    score, updated_at = state
    age_hours = max((timestamp - updated_at).total_seconds() / 3600.0, 0.0)
    value = score * 0.5 ** (age_hours / half_life_hours)
    return value if math.isfinite(value) else 0.0


def _update_prior_click_features(
    click_counts: Counter[str],
    decayed_scores: dict[str, tuple[float, datetime]],
    behavior: BehaviorImpression,
    half_life_hours: float,
) -> None:
    for candidate in behavior.candidates:
        if candidate.clicked != 1:
            continue
        click_counts[candidate.news_id] += 1
        current = _prior_decayed_score(
            decayed_scores,
            candidate.news_id,
            behavior.timestamp,
            half_life_hours,
        )
        decayed_scores[candidate.news_id] = (current + 1.0, behavior.timestamp)
