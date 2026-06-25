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
from feed_ranking_ops.retrieval.popularity import PopularityFallback
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig, build_user_profile
from feed_ranking_ops.retrieval.queries import RetrievalQuery
from feed_ranking_ops.retrieval.text import ArticleTextIndex


@dataclass(frozen=True)
class RetrievedArticle:
    news_id: str
    rank: int
    score: float
    was_in_history: bool


@dataclass
class RetrievalResult:
    query: RetrievalQuery
    retrieved: list[RetrievedArticle]
    target_news_ids: list[str]
    available_target_news_ids: list[str]
    unavailable_target_count: int
    missing_target_metadata_count: int
    catalog_size: int
    known_history_count: int
    unknown_history_count: int
    fallback_used: bool
    fallback_reason: str | None
    latency_seconds: float


def retrieve_for_query(
    query: RetrievalQuery,
    *,
    news: dict[str, NewsItem],
    article_index: ArticleTextIndex,
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

    profile = build_user_profile(query.history_news_ids, article_index, profile_config)
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
        ranked_ids, score_by_id = _exact_cosine_rank(
            eligible,
            article_index=article_index,
            availability=availability,
            profile=profile.vector,
        )

    retrieved = [
        RetrievedArticle(
            news_id=news_id,
            rank=rank,
            score=float(score_by_id[news_id]),
            was_in_history=news_id in set(query.history_news_ids),
        )
        for rank, news_id in enumerate(ranked_ids[:top_k], start=1)
    ]
    latency = perf_counter() - start
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
        latency_seconds=latency,
    )


def _exact_cosine_rank(
    eligible_news_ids: list[str],
    *,
    article_index: ArticleTextIndex,
    availability: ArticleAvailability,
    profile,
) -> tuple[list[str], dict[str, float]]:
    if profile is None:
        raise ValueError("profile must not be None for exact cosine ranking")
    row_ids = [
        article_index.article_to_row[news_id]
        for news_id in eligible_news_ids
        if news_id in article_index.article_to_row
    ]
    filtered_ids = [
        news_id for news_id in eligible_news_ids if news_id in article_index.article_to_row
    ]
    matrix = article_index.article_matrix[row_ids]
    profile_norm = float(np.sqrt(profile.multiply(profile).sum()))
    if matrix.shape[0] == 0 or profile_norm == 0.0:
        return [], {}
    raw_scores = matrix @ profile.T
    scores = np.asarray(raw_scores.toarray()).ravel() / profile_norm
    score_by_id = {
        news_id: float(score) for news_id, score in zip(filtered_ids, scores, strict=True)
    }
    ranked = sorted(
        filtered_ids,
        key=lambda news_id: (
            -score_by_id[news_id],
            availability.first_candidate_timestamp.get(news_id, datetime.max),
            news_id,
        ),
    )
    return ranked, score_by_id


def validate_retrieval_result(
    result: RetrievalResult,
    *,
    eligible_news_ids: set[str],
    exclude_history: bool,
) -> None:
    seen: set[str] = set()
    expected_rank = 1
    for item in result.retrieved:
        if item.news_id in seen:
            raise ValueError("duplicate retrieved article within query")
        seen.add(item.news_id)
        if item.news_id not in eligible_news_ids:
            raise ValueError("retrieved article is not in the eligible catalog")
        if item.rank != expected_rank:
            raise ValueError("retrieved ranks must be contiguous starting at 1")
        expected_rank += 1
        if exclude_history and item.was_in_history:
            raise ValueError("history article appeared in retrieval results")
