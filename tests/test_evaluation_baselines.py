from datetime import UTC, datetime, timedelta

import pytest

from feed_ranking_ops.evaluation.baselines import (
    CategoryAffinityBaseline,
    GlobalPopularityBaseline,
    OriginalOrderBaseline,
    TfidfContentSimilarityBaseline,
    TimeDecayedPopularityBaseline,
)
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)


def _news() -> dict[str, NewsItem]:
    return {
        "A": NewsItem("A", "sports", "soccer", "apple sports", "team apple"),
        "B": NewsItem("B", "sports", "soccer", "apple soccer", "goal"),
        "C": NewsItem("C", "finance", "markets", "quantum finance", "stocks"),
        "D": NewsItem("D", "health", "wellness", "nutrition", "wellness"),
    }


def _behavior(
    impression_id: str,
    *,
    hour: int = 0,
    history: list[str] | None = None,
    candidates: list[tuple[str, int | None]] | None = None,
) -> BehaviorImpression:
    return BehaviorImpression(
        partition="train",
        impression_id=impression_id,
        user_id="U",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        history_news_ids=history or [],
        candidates=[
            ImpressionCandidate(position=index, news_id=news_id, clicked=label)
            for index, (news_id, label) in enumerate(candidates or [("A", 1), ("B", 0)])
        ],
    )


def test_original_order_baseline_scores_source_order():
    baseline = OriginalOrderBaseline()
    behavior = _behavior("1", candidates=[("A", 0), ("B", 1), ("C", 0)])

    assert baseline.score(behavior, _news()).scores == [0.0, -1.0, -2.0]


def test_global_popularity_ranks_clicked_articles_and_fallbacks_unseen():
    baseline = GlobalPopularityBaseline()
    baseline.fit(
        [_behavior("1", candidates=[("A", 1), ("B", 0)]), _behavior("2", candidates=[("A", 1)])],
        _news(),
        fitting_partitions=["train"],
    )

    result = baseline.score(_behavior("3", candidates=[("A", 0), ("C", 0)]), _news())

    assert result.scores == [2.0, 0.0]
    assert result.metadata["unseen_candidate_count"] == 1
    assert baseline.metadata()["fitting_partitions"] == ["train"]


def test_time_decayed_popularity_uses_click_age():
    baseline = TimeDecayedPopularityBaseline(half_life_hours=24)
    baseline.fit(
        [
            _behavior("1", hour=0, candidates=[("A", 1)]),
            _behavior("2", hour=24, candidates=[("B", 1)]),
        ],
        _news(),
        fitting_partitions=["train"],
    )

    scores = baseline.score(_behavior("3", candidates=[("A", 0), ("B", 0)]), _news()).scores

    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1.0)


def test_category_affinity_ranks_matching_history_and_handles_empty_history():
    baseline = CategoryAffinityBaseline()
    baseline.fit([], _news(), fitting_partitions=["train"])
    behavior = _behavior("1", history=["A", "UNKNOWN"], candidates=[("B", 0), ("C", 1)])
    empty = _behavior("2", history=[], candidates=[("B", 0), ("C", 1)])

    result = baseline.score(behavior, _news())
    empty_result = baseline.score(empty, _news())

    assert result.scores[0] > result.scores[1]
    assert result.metadata["unknown_history_news_id_count"] == 1
    assert empty_result.scores == [0.0, 0.0]
    assert empty_result.metadata["empty_or_unknown_history_count"] == 1


def test_tfidf_similarity_ranks_content_and_handles_unseen_or_empty_history():
    news = _news()
    baseline = TfidfContentSimilarityBaseline()
    baseline.fit(
        [_behavior("fit", history=["A"], candidates=[("A", 1), ("B", 0)])],
        news,
        fitting_partitions=["train"],
    )

    result = baseline.score(
        _behavior("eval", history=["A"], candidates=[("B", 0), ("C", 1), ("D", 0)]),
        news,
    )
    empty = baseline.score(
        _behavior("empty", history=[], candidates=[("B", 0), ("C", 1)]),
        news,
    )

    assert result.scores[0] > result.scores[1]
    assert result.scores[2] == 0.0
    assert result.metadata["unseen_candidate_count"] == 2
    assert empty.scores == [0.0, 0.0]
    assert empty.metadata["empty_or_unseen_history_count"] == 1
