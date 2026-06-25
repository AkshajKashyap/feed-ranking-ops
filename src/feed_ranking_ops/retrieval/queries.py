from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from feed_ranking_ops.evaluation.processed import BehaviorImpression, NewsItem


@dataclass(frozen=True)
class RetrievalQuery:
    partition: str
    impression_id: str
    user_id: str
    timestamp: datetime
    history_news_ids: list[str]
    clicked_target_news_ids: list[str]
    impression_candidate_news_ids: list[str]


def behavior_to_retrieval_query(behavior: BehaviorImpression) -> RetrievalQuery:
    return RetrievalQuery(
        partition=behavior.partition,
        impression_id=behavior.impression_id,
        user_id=behavior.user_id,
        timestamp=behavior.timestamp,
        history_news_ids=list(behavior.history_news_ids),
        clicked_target_news_ids=[
            candidate.news_id for candidate in behavior.candidates if candidate.clicked == 1
        ],
        impression_candidate_news_ids=[
            candidate.news_id for candidate in behavior.candidates
        ],
    )


def behaviors_to_retrieval_queries(
    behaviors: list[BehaviorImpression],
) -> list[RetrievalQuery]:
    return [behavior_to_retrieval_query(behavior) for behavior in behaviors]


def query_target_diagnostics(
    queries: list[RetrievalQuery],
    news: dict[str, NewsItem],
) -> dict[str, Any]:
    missing_target_metadata = Counter()
    no_clicked_target = 0
    for query in queries:
        if not query.clicked_target_news_ids:
            no_clicked_target += 1
        for news_id in query.clicked_target_news_ids:
            if news_id not in news:
                missing_target_metadata[query.partition] += 1
    return {
        "query_count": len(queries),
        "queries_with_no_clicked_target": no_clicked_target,
        "missing_clicked_target_metadata_count": sum(missing_target_metadata.values()),
        "missing_clicked_target_metadata_by_partition": dict(
            sorted(missing_target_metadata.items())
        ),
    }
