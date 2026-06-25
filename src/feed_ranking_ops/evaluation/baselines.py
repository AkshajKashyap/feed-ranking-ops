from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    NewsItem,
)


@dataclass(frozen=True)
class BaselineScoreResult:
    scores: list[float]
    metadata: dict[str, int] = field(default_factory=dict)


class RankingBaseline:
    name: str = "baseline"

    def config(self) -> dict[str, Any]:
        return {}

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> None:
        self.fitted_metadata = {
            "fitting_partitions": list(fitting_partitions),
            "fitting_impression_count": len(behaviors),
        }

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return getattr(self, "fitted_metadata", {})


class OriginalOrderBaseline(RankingBaseline):
    name = "original_order"

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        del news
        return BaselineScoreResult(
            scores=[-float(candidate.position) for candidate in behavior.candidates]
        )


class GlobalPopularityBaseline(RankingBaseline):
    name = "global_popularity"

    def __init__(self, fallback_score: float = 0.0) -> None:
        self.fallback_score = fallback_score
        self.click_counts: Counter[str] = Counter()

    def config(self) -> dict[str, Any]:
        return {"fallback_score": self.fallback_score}

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> None:
        super().fit(behaviors, news, fitting_partitions=fitting_partitions)
        self.click_counts = Counter(
            candidate.news_id
            for behavior in behaviors
            for candidate in behavior.candidates
            if candidate.clicked == 1
        )
        self.fitted_metadata.update(
            {
                "uses_labels": True,
                "positive_click_events": sum(self.click_counts.values()),
                "clicked_article_count": len(self.click_counts),
            }
        )

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        del news
        unseen = sum(
            1 for candidate in behavior.candidates if candidate.news_id not in self.click_counts
        )
        return BaselineScoreResult(
            scores=[
                float(self.click_counts.get(candidate.news_id, self.fallback_score))
                for candidate in behavior.candidates
            ],
            metadata={"unseen_candidate_count": unseen},
        )


class TimeDecayedPopularityBaseline(RankingBaseline):
    name = "time_decayed_popularity"

    def __init__(self, half_life_hours: float = 24.0, fallback_score: float = 0.0) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self.half_life_hours = half_life_hours
        self.fallback_score = fallback_score
        self.scores: Counter[str] = Counter()
        self.cutoff: datetime | None = None

    @property
    def selection_name(self) -> str:
        return f"{self.name}_{self.half_life_hours:g}h"

    def config(self) -> dict[str, Any]:
        return {
            "half_life_hours": self.half_life_hours,
            "fallback_score": self.fallback_score,
            "decay": "0.5 ** (age_hours / half_life_hours)",
        }

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> None:
        super().fit(behaviors, news, fitting_partitions=fitting_partitions)
        del news
        self.cutoff = max((behavior.timestamp for behavior in behaviors), default=None)
        self.scores = Counter()
        if self.cutoff is not None:
            for behavior in behaviors:
                age_hours = max(
                    (self.cutoff - behavior.timestamp).total_seconds() / 3600.0,
                    0.0,
                )
                weight = 0.5 ** (age_hours / self.half_life_hours)
                for candidate in behavior.candidates:
                    if candidate.clicked == 1:
                        self.scores[candidate.news_id] += weight
        self.fitted_metadata.update(
            {
                "uses_labels": True,
                "positive_click_events": sum(
                    1
                    for behavior in behaviors
                    for candidate in behavior.candidates
                    if candidate.clicked == 1
                ),
                "decayed_clicked_article_count": len(self.scores),
                "cutoff_timestamp": self.cutoff.isoformat() if self.cutoff else None,
                "half_life_hours": self.half_life_hours,
            }
        )

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        del news
        unseen = sum(1 for candidate in behavior.candidates if candidate.news_id not in self.scores)
        return BaselineScoreResult(
            scores=[
                float(self.scores.get(candidate.news_id, self.fallback_score))
                for candidate in behavior.candidates
            ],
            metadata={"unseen_candidate_count": unseen},
        )


class CategoryAffinityBaseline(RankingBaseline):
    name = "category_affinity"

    def __init__(
        self,
        *,
        category_weight: float = 1.0,
        subcategory_weight: float = 0.5,
        fallback_score: float = 0.0,
    ) -> None:
        self.category_weight = category_weight
        self.subcategory_weight = subcategory_weight
        self.fallback_score = fallback_score

    def config(self) -> dict[str, Any]:
        return {
            "category_weight": self.category_weight,
            "subcategory_weight": self.subcategory_weight,
            "fallback_score": self.fallback_score,
        }

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> None:
        super().fit(behaviors, news, fitting_partitions=fitting_partitions)
        self.fitted_metadata.update(
            {
                "uses_labels": False,
                "uses_current_impression_labels": False,
                "known_news_count": len(news),
            }
        )

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        category_counts: Counter[str] = Counter()
        subcategory_counts: Counter[str] = Counter()
        unknown_history = 0
        for news_id in behavior.history_news_ids:
            item = news.get(news_id)
            if item is None:
                unknown_history += 1
                continue
            category_counts[item.category] += 1
            subcategory_counts[item.subcategory] += 1

        if not category_counts and not subcategory_counts:
            return BaselineScoreResult(
                scores=[self.fallback_score for _ in behavior.candidates],
                metadata={
                    "empty_or_unknown_history_count": 1,
                    "unknown_history_news_id_count": unknown_history,
                },
            )

        scores = []
        for candidate in behavior.candidates:
            item = news[candidate.news_id]
            scores.append(
                float(
                    self.category_weight * category_counts[item.category]
                    + self.subcategory_weight * subcategory_counts[item.subcategory]
                )
            )
        return BaselineScoreResult(
            scores=scores,
            metadata={"unknown_history_news_id_count": unknown_history},
        )


class TfidfContentSimilarityBaseline(RankingBaseline):
    name = "tfidf_content_similarity"

    def __init__(self, fallback_score: float = 0.0) -> None:
        self.fallback_score = fallback_score
        self.vectorizer: TfidfVectorizer | None = None
        self.article_index: dict[str, int] = {}
        self.article_matrix = None

    def config(self) -> dict[str, Any]:
        return {
            "fallback_score": self.fallback_score,
            "text_fields": ["title", "abstract"],
            "vocabulary_policy": (
                "Fit vocabulary on news articles referenced by the allowed fitting "
                "partition histories or candidates."
            ),
        }

    def fit(
        self,
        behaviors: list[BehaviorImpression],
        news: dict[str, NewsItem],
        *,
        fitting_partitions: list[str],
    ) -> None:
        super().fit(behaviors, news, fitting_partitions=fitting_partitions)
        observed_news_ids = sorted(
            {
                news_id
                for behavior in behaviors
                for news_id in [
                    *behavior.history_news_ids,
                    *(candidate.news_id for candidate in behavior.candidates),
                ]
                if news_id in news
            }
        )
        texts = [news[news_id].text for news_id in observed_news_ids]
        non_empty_texts = [text for text in texts if text.strip()]
        self.article_index = {}
        self.article_matrix = None
        self.vectorizer = None
        if non_empty_texts:
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(non_empty_texts)
            self.article_matrix = self.vectorizer.transform(texts)
            self.article_index = {
                news_id: index for index, news_id in enumerate(observed_news_ids)
            }
        self.fitted_metadata.update(
            {
                "uses_labels": False,
                "observed_article_count": len(observed_news_ids),
                "vocabulary_size": (
                    len(self.vectorizer.vocabulary_) if self.vectorizer is not None else 0
                ),
            }
        )

    def score(
        self,
        behavior: BehaviorImpression,
        news: dict[str, NewsItem],
    ) -> BaselineScoreResult:
        del news
        if self.vectorizer is None or self.article_matrix is None:
            return BaselineScoreResult(
                scores=[self.fallback_score for _ in behavior.candidates],
                metadata={"empty_vocabulary_count": 1},
            )
        history_indices = [
            self.article_index[news_id]
            for news_id in behavior.history_news_ids
            if news_id in self.article_index
        ]
        unknown_history = len(behavior.history_news_ids) - len(history_indices)
        if not history_indices:
            return BaselineScoreResult(
                scores=[self.fallback_score for _ in behavior.candidates],
                metadata={
                    "empty_or_unseen_history_count": 1,
                    "unknown_history_news_id_count": unknown_history,
                },
            )

        profile = np.asarray(self.article_matrix[history_indices].mean(axis=0)).ravel()
        profile_norm = float(np.linalg.norm(profile))
        if not math.isfinite(profile_norm) or profile_norm == 0.0:
            return BaselineScoreResult(
                scores=[self.fallback_score for _ in behavior.candidates],
                metadata={"all_zero_user_vector_count": 1},
            )

        scores = []
        unseen_candidates = 0
        for candidate in behavior.candidates:
            index = self.article_index.get(candidate.news_id)
            if index is None:
                unseen_candidates += 1
                scores.append(self.fallback_score)
                continue
            raw_score = self.article_matrix[index].dot(profile) / profile_norm
            score = float(np.asarray(raw_score).ravel()[0])
            scores.append(score if math.isfinite(score) else self.fallback_score)
        return BaselineScoreResult(
            scores=scores,
            metadata={
                "unseen_candidate_count": unseen_candidates,
                "unknown_history_news_id_count": unknown_history,
            },
        )


def make_baselines(
    names: list[str],
    *,
    half_life_hours: list[float],
) -> list[RankingBaseline]:
    factories = {
        "original_order": lambda: [OriginalOrderBaseline()],
        "global_popularity": lambda: [GlobalPopularityBaseline()],
        "time_decayed_popularity": lambda: [
            TimeDecayedPopularityBaseline(half_life_hours=value)
            for value in half_life_hours
        ],
        "category_affinity": lambda: [CategoryAffinityBaseline()],
        "tfidf_content_similarity": lambda: [TfidfContentSimilarityBaseline()],
    }
    baselines: list[RankingBaseline] = []
    invalid = sorted(set(names).difference(factories))
    if invalid:
        raise ValueError(f"Unknown baselines: {', '.join(invalid)}")
    for name in names:
        baselines.extend(factories[name]())
    return baselines


def default_baseline_names() -> list[str]:
    return [
        "original_order",
        "global_popularity",
        "time_decayed_popularity",
        "category_affinity",
        "tfidf_content_similarity",
    ]
