from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np

from feed_ranking_ops.evaluation.baselines import TfidfContentSimilarityBaseline
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)
from feed_ranking_ops.ranking.features import CandidateFeatureBuilder, FEATURE_NAMES


def _news() -> dict[str, NewsItem]:
    return {
        "N1": NewsItem("N1", "news", "local", "Local title", "Local abstract"),
        "N2": NewsItem("N2", "sports", "football", "Football title", ""),
        "N3": NewsItem("N3", "news", "local", "Another local", "More text"),
    }


def _behaviors() -> list[BehaviorImpression]:
    timestamp = datetime(2019, 11, 15, 8, tzinfo=UTC)
    return [
        BehaviorImpression(
            partition="train",
            impression_id="1",
            user_id="U1",
            timestamp=timestamp,
            history_news_ids=[],
            candidates=[
                ImpressionCandidate(0, "N1", 1),
                ImpressionCandidate(1, "N2", 0),
            ],
        ),
        BehaviorImpression(
            partition="train",
            impression_id="2",
            user_id="U2",
            timestamp=timestamp + timedelta(hours=1),
            history_news_ids=["N1", "MISSING"],
            candidates=[
                ImpressionCandidate(0, "N1", 0),
                ImpressionCandidate(1, "N3", 1),
            ],
        ),
    ]


def test_candidate_explosion_and_feature_matrix_shape():
    news = _news()
    behaviors = _behaviors()
    matrix = CandidateFeatureBuilder().fit(
        behaviors,
        news,
        fitting_partitions=["train"],
    ).transform(behaviors, news, chronological_training=True)

    assert matrix.features.shape == (4, len(FEATURE_NAMES))
    assert matrix.labels.tolist() == [1, 0, 0, 1]
    assert matrix.offsets.tolist() == [0, 2, 4]
    assert matrix.n_candidates == 4


def test_training_popularity_excludes_current_impression_labels():
    news = _news()
    behaviors = _behaviors()
    builder = CandidateFeatureBuilder().fit(
        behaviors,
        news,
        fitting_partitions=["train"],
    )
    original = builder.transform(behaviors, news, chronological_training=True)
    changed_first = replace(
        behaviors[0],
        candidates=[
            replace(behaviors[0].candidates[0], clicked=0),
            replace(behaviors[0].candidates[1], clicked=1),
        ],
    )
    changed = builder.transform(
        [changed_first, behaviors[1]],
        news,
        chronological_training=True,
    )

    np.testing.assert_array_equal(original.features[:2], changed.features[:2])
    assert original.features[0, FEATURE_NAMES.index("global_click_count")] == 0
    assert original.features[1, FEATURE_NAMES.index("global_click_count")] == 0


def test_evaluation_features_do_not_depend_on_current_labels():
    news = _news()
    fit_behaviors = _behaviors()
    evaluation = replace(
        fit_behaviors[1],
        partition="validation",
        candidates=[
            replace(fit_behaviors[1].candidates[0], clicked=1),
            replace(fit_behaviors[1].candidates[1], clicked=0),
        ],
    )
    changed = replace(
        evaluation,
        candidates=[
            replace(evaluation.candidates[0], clicked=0),
            replace(evaluation.candidates[1], clicked=1),
        ],
    )
    builder = CandidateFeatureBuilder().fit(
        fit_behaviors,
        news,
        fitting_partitions=["train"],
    )

    first = builder.transform([evaluation], news)
    second = builder.transform([changed], news)

    np.testing.assert_array_equal(first.features, second.features)
    assert first.metadata["fitting_partitions"] == ["train"]
    assert first.metadata["current_impression_labels_used"] is False


def test_empty_and_partially_missing_history_features_are_explicit():
    news = _news()
    behaviors = _behaviors()
    matrix = CandidateFeatureBuilder().fit(
        behaviors,
        news,
        fitting_partitions=["train"],
    ).transform(behaviors, news, chronological_training=True)
    empty_index = FEATURE_NAMES.index("empty_history")
    known_index = FEATURE_NAMES.index("known_history_count")

    assert matrix.features[0, empty_index] == 1
    assert matrix.features[2, empty_index] == 0
    assert matrix.features[2, known_index] == 1


def test_batched_tfidf_feature_matches_existing_baseline():
    news = _news()
    behaviors = _behaviors()
    matrix = CandidateFeatureBuilder().fit(
        behaviors,
        news,
        fitting_partitions=["train"],
    ).transform(behaviors, news)
    baseline = TfidfContentSimilarityBaseline()
    baseline.fit(behaviors, news, fitting_partitions=["train"])
    expected = [
        score
        for behavior in behaviors
        for score in baseline.score(behavior, news).scores
    ]

    np.testing.assert_allclose(
        matrix.features[:, FEATURE_NAMES.index("tfidf_history_similarity")],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
