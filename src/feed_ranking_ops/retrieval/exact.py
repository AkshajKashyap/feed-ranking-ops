from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

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


@dataclass(frozen=True)
class PreparedQueryRetrieval:
    eligible_news_ids: list[str]
    eligible_row_ids: np.ndarray
    target_info: dict[str, object]


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
    prepared_query: PreparedQueryRetrieval | None = None,
) -> RetrievalResult:
    start = perf_counter()
    if prepared_query is None:
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
        eligible_row_ids = None
    else:
        target_info = prepared_query.target_info
        eligible = prepared_query.eligible_news_ids
        eligible_row_ids = prepared_query.eligible_row_ids

    profile = build_user_profile(query.history_news_ids, article_index, profile_config)
    fallback_reason = profile.fallback_reason
    fallback_used = fallback_reason is not None
    if not eligible:
        ranked_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        fallback_used = True
        fallback_reason = fallback_reason or "no_eligible_articles"
    elif fallback_used:
        ranked_ids = fallback.rank(eligible, availability, top_k=top_k)
        score_by_id = {news_id: fallback.score(news_id) for news_id in ranked_ids}
    else:
        ranked_ids, score_by_id = _exact_cosine_rank(
            eligible,
            article_index=article_index,
            availability=availability,
            profile=profile.vector,
            eligible_row_ids=eligible_row_ids,
            top_k=top_k,
        )

    history = set(query.history_news_ids)
    retrieved = [
        RetrievedArticle(
            news_id=news_id,
            rank=rank,
            score=float(score_by_id[news_id]),
            was_in_history=news_id in history,
        )
        for rank, news_id in enumerate(ranked_ids, start=1)
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
    eligible_row_ids: np.ndarray | None = None,
    top_k: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    del availability
    if profile is None:
        raise ValueError("profile must not be None for exact cosine ranking")
    if eligible_row_ids is None:
        filtered = [
            (news_id, article_index.article_to_row[news_id])
            for news_id in eligible_news_ids
            if news_id in article_index.article_to_row
        ]
        filtered_ids = [news_id for news_id, _ in filtered]
        row_ids = np.asarray([row_id for _, row_id in filtered], dtype=np.int64)
    else:
        filtered_ids = eligible_news_ids
        row_ids = eligible_row_ids
    matrix = article_index.article_matrix[row_ids]
    profile_norm = float(np.sqrt(profile.multiply(profile).sum()))
    if matrix.shape[0] == 0 or profile_norm == 0.0:
        return [], {}
    raw_scores = matrix @ profile.T
    scores = np.asarray(raw_scores.toarray()).ravel() / profile_norm
    selected_indices = _top_score_indices(scores, top_k)
    ranked = [filtered_ids[index] for index in selected_indices]
    score_by_id = {
        filtered_ids[index]: float(scores[index]) for index in selected_indices
    }
    return ranked, score_by_id


def _top_score_indices(scores: np.ndarray, top_k: int | None) -> list[int]:
    # Eligible IDs arrive in availability/article-ID order, so source index is
    # the deterministic secondary key for equal scores.
    count = len(scores)
    if top_k is None or top_k >= count:
        selected = np.arange(count)
    elif top_k <= 0:
        return []
    else:
        partition = np.argpartition(scores, -top_k)[-top_k:]
        threshold = float(scores[partition].min())
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        selected = np.concatenate((above, tied[: top_k - len(above)]))
    return sorted(
        (int(index) for index in selected),
        key=lambda index: (-float(scores[index]), index),
    )


def validate_retrieval_result(
    result: RetrievalResult,
    *,
    eligible_news_ids: set[str] | None = None,
    eligibility_check: Callable[[str], bool] | None = None,
    exclude_history: bool,
) -> None:
    if eligible_news_ids is None and eligibility_check is None:
        raise ValueError("eligible_news_ids or eligibility_check is required")
    seen: set[str] = set()
    expected_rank = 1
    for item in result.retrieved:
        if item.news_id in seen:
            raise ValueError("duplicate retrieved article within query")
        seen.add(item.news_id)
        is_eligible = (
            item.news_id in eligible_news_ids
            if eligible_news_ids is not None
            else eligibility_check is not None and eligibility_check(item.news_id)
        )
        if not is_eligible:
            raise ValueError("retrieved article is not in the eligible catalog")
        if item.rank != expected_rank:
            raise ValueError("retrieved ranks must be contiguous starting at 1")
        expected_rank += 1
        if exclude_history and item.was_in_history:
            raise ValueError("history article appeared in retrieval results")
