from datetime import UTC, datetime, timedelta

import pytest

from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)
from feed_ranking_ops.retrieval.availability import (
    derive_article_availability,
    eligible_catalog,
    static_catalog_from_partitions,
)
from feed_ranking_ops.retrieval.exact import retrieve_for_query
from feed_ranking_ops.retrieval.metrics import evaluate_retrieval_results
from feed_ranking_ops.retrieval.popularity import fit_popularity_fallback
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig, build_user_profile
from feed_ranking_ops.retrieval.queries import behavior_to_retrieval_query
from feed_ranking_ops.retrieval.text import TextConfig, build_article_text, fit_article_text_index


def _news() -> dict[str, NewsItem]:
    return {
        "A": NewsItem("A", "sports", "soccer", "apple match", "team wins"),
        "B": NewsItem("B", "sports", "soccer", "apple goal", "match recap"),
        "C": NewsItem("C", "finance", "markets", "bond market", "rates"),
        "D": NewsItem("D", "tech", "ai", "quantum device", "launch"),
    }


def _behavior(
    impression_id: str,
    *,
    hour: int,
    history: list[str],
    candidates: list[tuple[str, int | None]],
    partition: str = "train",
) -> BehaviorImpression:
    return BehaviorImpression(
        partition=partition,
        impression_id=impression_id,
        user_id="U",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        history_news_ids=history,
        candidates=[
            ImpressionCandidate(index, news_id, label)
            for index, (news_id, label) in enumerate(candidates)
        ],
    )


def test_availability_excludes_future_and_includes_equal_timestamp():
    behaviors = {
        "train": [
            _behavior("1", hour=1, history=[], candidates=[("A", 0)]),
            _behavior("2", hour=2, history=["A"], candidates=[("B", 1)]),
            _behavior("3", hour=3, history=[], candidates=[("C", 1)]),
        ]
    }
    availability = derive_article_availability(behaviors)
    query = behavior_to_retrieval_query(behaviors["train"][1])

    eligible = eligible_catalog(
        query,
        news=_news(),
        availability=availability,
        protocol="observed_available",
    )

    assert eligible == ["A", "B"]
    assert "C" not in eligible


def test_static_catalog_can_include_future_observed_articles():
    behaviors = [
        _behavior("1", hour=1, history=[], candidates=[("A", 0)]),
        _behavior("2", hour=3, history=[], candidates=[("C", 1)]),
    ]
    availability = derive_article_availability({"train": behaviors})
    static = static_catalog_from_partitions(behaviors, _news())

    eligible = eligible_catalog(
        behavior_to_retrieval_query(behaviors[0]),
        news=_news(),
        availability=availability,
        protocol="static_partition_catalog",
        static_catalog_ids=static,
    )

    assert eligible == ["A", "C"]


def test_query_creation_preserves_multiple_targets_and_history_order():
    behavior = _behavior(
        "1",
        hour=1,
        history=["A", "B", "A"],
        candidates=[("C", 1), ("D", 0), ("B", 1)],
    )

    query = behavior_to_retrieval_query(behavior)

    assert query.history_news_ids == ["A", "B", "A"]
    assert query.clicked_target_news_ids == ["C", "B"]
    assert query.impression_candidate_news_ids == ["C", "D", "B"]


def test_article_text_configurations():
    item = _news()["A"]

    assert build_article_text(item, TextConfig("title")) == "apple match"
    assert "team wins" in build_article_text(item, TextConfig("title_abstract"))
    assert "category_sports" in build_article_text(
        item,
        TextConfig("title_abstract_category"),
    )


def test_mean_and_recency_profiles_with_truncation_and_unknowns():
    news = _news()
    fit = [_behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])]
    index = fit_article_text_index(
        news=news,
        fitting_behaviors=fit,
        text_config=TextConfig("title_abstract"),
    )

    mean_profile = build_user_profile(
        ["UNKNOWN", "A", "B"],
        index,
        HistoryProfileConfig("mean", max_history_length=2),
    )
    recency_profile = build_user_profile(
        ["A", "B"],
        index,
        HistoryProfileConfig("recency", max_history_length=None, decay=0.5),
    )
    empty = build_user_profile([], index, HistoryProfileConfig("mean"))

    assert mean_profile.known_history_count == 2
    assert mean_profile.unknown_history_count == 0
    assert recency_profile.vector is not None
    assert empty.fallback_reason == "empty_history"


def test_exact_retrieval_excludes_history_and_uses_deterministic_scores():
    news = _news()
    fit = [
        _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)]),
        _behavior("eval", hour=2, history=["A"], candidates=[("B", 1), ("D", 0)]),
    ]
    availability = derive_article_availability({"train": fit})
    index = fit_article_text_index(
        news=news,
        fitting_behaviors=[fit[0]],
        text_config=TextConfig("title_abstract"),
    )
    fallback = fit_popularity_fallback([fit[0]], fitting_partitions=["train"])

    result = retrieve_for_query(
        behavior_to_retrieval_query(fit[1]),
        news=news,
        article_index=index,
        availability=availability,
        fallback=fallback,
        profile_config=HistoryProfileConfig("mean"),
        catalog_protocol="observed_available",
        static_catalog_ids=None,
        top_k=10,
        exclude_history=True,
    )

    assert all(item.news_id != "A" for item in result.retrieved)
    assert result.retrieved[0].news_id == "B"
    assert len({item.news_id for item in result.retrieved}) == len(result.retrieved)
    assert [item.rank for item in result.retrieved] == list(range(1, len(result.retrieved) + 1))


def test_popularity_fallback_for_empty_history_records_provenance():
    news = _news()
    fit = [_behavior("fit", hour=1, history=[], candidates=[("C", 1), ("B", 0)])]
    eval_behavior = _behavior("eval", hour=2, history=[], candidates=[("B", 0), ("C", 1)])
    availability = derive_article_availability({"train": [*fit, eval_behavior]})
    index = fit_article_text_index(
        news=news,
        fitting_behaviors=fit,
        text_config=TextConfig("title"),
    )
    fallback = fit_popularity_fallback(fit, fitting_partitions=["train"])

    result = retrieve_for_query(
        behavior_to_retrieval_query(eval_behavior),
        news=news,
        article_index=index,
        availability=availability,
        fallback=fallback,
        profile_config=HistoryProfileConfig("mean"),
        catalog_protocol="observed_available",
        static_catalog_ids=None,
        top_k=10,
        exclude_history=True,
    )

    assert result.fallback_used is True
    assert result.fallback_reason == "empty_history"
    assert result.retrieved[0].news_id == "C"
    assert fallback.metadata()["fitting_partitions"] == ["train"]


def test_retrieval_metrics_multiple_targets_and_availability():
    news = _news()
    fit = [
        _behavior("fit", hour=1, history=["A"], candidates=[("A", 0), ("B", 1), ("C", 1)]),
    ]
    availability = derive_article_availability({"train": fit})
    index = fit_article_text_index(
        news=news,
        fitting_behaviors=fit,
        text_config=TextConfig("title_abstract"),
    )
    fallback = fit_popularity_fallback(fit, fitting_partitions=["train"])
    result = retrieve_for_query(
        behavior_to_retrieval_query(fit[0]),
        news=news,
        article_index=index,
        availability=availability,
        fallback=fallback,
        profile_config=HistoryProfileConfig("mean"),
        catalog_protocol="observed_available",
        static_catalog_ids=None,
        top_k=100,
        exclude_history=True,
    )

    metrics = evaluate_retrieval_results([result], catalog_size_total=len(news))

    assert metrics["n_valid_queries"] == 1
    assert metrics["metrics"]["recall@10"] == pytest.approx(1.0)
    assert metrics["metrics"]["hit_rate@10"] == pytest.approx(1.0)
    assert metrics["metrics"]["mrr@100"] is not None
    assert metrics["metrics"]["ndcg@10"] is not None
