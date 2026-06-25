from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from feed_ranking_ops.evaluation.processed import BehaviorImpression, NewsItem
from feed_ranking_ops.retrieval.queries import RetrievalQuery

CatalogProtocol = Literal["observed_available", "static_partition_catalog"]


@dataclass(frozen=True)
class ArticleAvailability:
    first_candidate_timestamp: dict[str, datetime]
    first_history_timestamp: dict[str, datetime]


def derive_article_availability(
    behaviors_by_partition: dict[str, list[BehaviorImpression]],
) -> ArticleAvailability:
    first_candidate: dict[str, datetime] = {}
    first_history: dict[str, datetime] = {}
    for behaviors in behaviors_by_partition.values():
        for behavior in behaviors:
            for candidate in behavior.candidates:
                _record_first(first_candidate, candidate.news_id, behavior.timestamp)
            for news_id in behavior.history_news_ids:
                _record_first(first_history, news_id, behavior.timestamp)
    return ArticleAvailability(
        first_candidate_timestamp=first_candidate,
        first_history_timestamp=first_history,
    )


def eligible_catalog(
    query: RetrievalQuery,
    *,
    news: dict[str, NewsItem],
    availability: ArticleAvailability,
    protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None = None,
) -> list[str]:
    if protocol == "observed_available":
        eligible = [
            news_id
            for news_id, timestamp in availability.first_candidate_timestamp.items()
            if timestamp <= query.timestamp and news_id in news
        ]
    elif protocol == "static_partition_catalog":
        if static_catalog_ids is None:
            raise ValueError("static_catalog_ids is required for static_partition_catalog")
        eligible = [news_id for news_id in static_catalog_ids if news_id in news]
    else:
        raise ValueError(f"Unknown catalog protocol: {protocol}")
    return sorted(
        eligible,
        key=lambda news_id: (
            availability.first_candidate_timestamp.get(news_id, datetime.max),
            news_id,
        ),
    )


def static_catalog_from_partitions(
    behaviors: list[BehaviorImpression],
    news: dict[str, NewsItem],
) -> set[str]:
    return {
        candidate.news_id
        for behavior in behaviors
        for candidate in behavior.candidates
        if candidate.news_id in news
    }


def target_availability(
    query: RetrievalQuery,
    *,
    news: dict[str, NewsItem],
    availability: ArticleAvailability,
    protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None = None,
) -> dict[str, object]:
    targets_with_metadata = [
        news_id for news_id in query.clicked_target_news_ids if news_id in news
    ]
    eligible = set(
        eligible_catalog(
            query,
            news=news,
            availability=availability,
            protocol=protocol,
            static_catalog_ids=static_catalog_ids,
        )
    )
    available = [news_id for news_id in targets_with_metadata if news_id in eligible]
    unavailable = [
        news_id for news_id in targets_with_metadata if news_id not in eligible
    ]
    return {
        "targets_with_metadata": targets_with_metadata,
        "available_targets": available,
        "unavailable_targets": unavailable,
        "missing_metadata_count": len(query.clicked_target_news_ids)
        - len(targets_with_metadata),
    }


def _record_first(
    values: dict[str, datetime],
    news_id: str,
    timestamp: datetime,
) -> None:
    existing = values.get(news_id)
    if existing is None or timestamp < existing:
        values[news_id] = timestamp
