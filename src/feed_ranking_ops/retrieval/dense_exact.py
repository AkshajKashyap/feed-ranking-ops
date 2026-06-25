from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

import numpy as np

from feed_ranking_ops.evaluation.processed import NewsItem
from feed_ranking_ops.retrieval.availability import (
    ArticleAvailability,
    CatalogProtocol,
    eligible_catalog,
    target_availability,
)
from feed_ranking_ops.retrieval.dense import DenseArticleIndex, build_dense_user_profile
from feed_ranking_ops.retrieval.exact import RetrievedArticle, RetrievalResult
from feed_ranking_ops.retrieval.popularity import PopularityFallback
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.queries import RetrievalQuery


@dataclass(frozen=True)
class DenseExactRankDiagnostics:
    candidates_scored: int


def retrieve_dense_exact_for_query(
    query: RetrievalQuery,
    *,
    news: dict[str, NewsItem],
    dense_index: DenseArticleIndex,
    availability: ArticleAvailability,
    fallback: PopularityFallback,
    profile_config: HistoryProfileConfig,
    catalog_protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None,
    top_k: int,
    exclude_history: bool,
) -> RetrievalResult:
    start = perf_counter()
    target_info = target_availability(
        query,
        news=news,
        availability=availability,
        protocol=catalog_protocol,
        static_catalog_ids=static_catalog_ids,
    )
    eligible = eligible_catalog(
        query,
        news=news,
        availability=availability,
        protocol=catalog_protocol,
        static_catalog_ids=static_catalog_ids,
    )
    if exclude_history:
        history = set(query.history_news_ids)
        eligible = [news_id for news_id in eligible if news_id not in history]

    profile = build_dense_user_profile(query.history_news_ids, dense_index, profile_config)
    fallback_reason = profile.fallback_reason
    fallback_used = fallback_reason is not None
    if not eligible:
        ranked_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        fallback_used = True
        fallback_reason = fallback_reason or "no_eligible_articles"
    elif fallback_used:
        ranked_ids = fallback.rank(eligible, availability)
        score_by_id = {news_id: fallback.score(news_id) for news_id in ranked_ids}
    else:
        ranked_ids, score_by_id, _diagnostics = dense_exact_rank(
            eligible,
            dense_index=dense_index,
            availability=availability,
            query_vector=profile.vector,
        )

    history_ids = set(query.history_news_ids)
    retrieved = [
        RetrievedArticle(
            news_id=news_id,
            rank=rank,
            score=float(score_by_id[news_id]),
            was_in_history=news_id in history_ids,
        )
        for rank, news_id in enumerate(ranked_ids[:top_k], start=1)
    ]
    return RetrievalResult(
        query=query,
        retrieved=retrieved,
        target_news_ids=list(query.clicked_target_news_ids),
        available_target_news_ids=list(target_info["available_targets"]),
        unavailable_target_count=len(target_info["unavailable_targets"]),
        missing_target_metadata_count=int(target_info["missing_metadata_count"]),
        catalog_size=len(eligible),
        known_history_count=profile.known_history_count,
        unknown_history_count=profile.unknown_history_count,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        latency_seconds=perf_counter() - start,
    )


def dense_exact_rank(
    eligible_news_ids: list[str],
    *,
    dense_index: DenseArticleIndex,
    availability: ArticleAvailability,
    query_vector: np.ndarray | None,
) -> tuple[list[str], dict[str, float], DenseExactRankDiagnostics]:
    if query_vector is None:
        raise ValueError("query_vector must not be None for dense exact ranking")
    filtered_ids = [
        news_id for news_id in eligible_news_ids if news_id in dense_index.article_to_row
    ]
    if not filtered_ids:
        return [], {}, DenseExactRankDiagnostics(candidates_scored=0)
    rows = [dense_index.article_to_row[news_id] for news_id in filtered_ids]
    matrix = dense_index.vectors[rows]
    scores = matrix @ np.asarray(query_vector, dtype=np.float32)
    score_by_id = {
        news_id: float(score)
        for news_id, score in zip(filtered_ids, scores.tolist(), strict=True)
    }
    ranked = sorted(
        filtered_ids,
        key=lambda news_id: (
            -score_by_id[news_id],
            availability.first_candidate_timestamp.get(news_id, datetime.max),
            news_id,
        ),
    )
    return ranked, score_by_id, DenseExactRankDiagnostics(candidates_scored=len(filtered_ids))
