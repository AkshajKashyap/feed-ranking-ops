from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from feed_ranking_ops.evaluation.processed import BehaviorImpression
from feed_ranking_ops.retrieval.availability import ArticleAvailability


@dataclass
class PopularityFallback:
    click_counts: Counter[str]
    fitting_partitions: list[str]

    def score(self, news_id: str) -> float:
        return float(self.click_counts.get(news_id, 0))

    def rank(
        self,
        eligible_news_ids: list[str],
        availability: ArticleAvailability,
        *,
        top_k: int | None = None,
    ) -> list[str]:
        def key(news_id: str) -> tuple[float, datetime, str]:
            return (
                -self.score(news_id),
                availability.first_candidate_timestamp.get(news_id, datetime.max),
                news_id,
            )

        if top_k is not None and top_k < len(eligible_news_ids):
            return heapq.nsmallest(top_k, eligible_news_ids, key=key)
        return sorted(eligible_news_ids, key=key)

    def metadata(self) -> dict[str, object]:
        return {
            "fitting_partitions": list(self.fitting_partitions),
            "positive_click_events": sum(self.click_counts.values()),
            "clicked_article_count": len(self.click_counts),
            "uses_labels": True,
        }


def fit_popularity_fallback(
    behaviors: list[BehaviorImpression],
    *,
    fitting_partitions: list[str],
) -> PopularityFallback:
    counts = Counter(
        candidate.news_id
        for behavior in behaviors
        for candidate in behavior.candidates
        if candidate.clicked == 1
    )
    return PopularityFallback(click_counts=counts, fitting_partitions=fitting_partitions)
