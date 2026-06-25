import math

import pytest

from feed_ranking_ops.evaluation.metrics import (
    ScoredCandidate,
    aggregate_scored_candidates,
    deterministic_rank_order,
    evaluate_scored_candidates,
)


def _candidate(position: int, label: int | None, score: float | None) -> ScoredCandidate:
    return ScoredCandidate(
        impression_id="imp",
        candidate_position=position,
        click_label=label,
        score=score,
    )


def test_metrics_match_manually_computed_example():
    candidates = [
        _candidate(0, 1, 0.2),
        _candidate(1, 0, 0.9),
        _candidate(2, 1, 0.1),
    ]

    result = evaluate_scored_candidates(candidates)

    ideal_dcg = 1 + 1 / math.log2(3)
    observed_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    assert result.metrics["mrr"] == pytest.approx(0.5)
    assert result.metrics["ndcg@5"] == pytest.approx(observed_dcg / ideal_dcg)
    assert result.metrics["recall@5"] == pytest.approx(1.0)
    assert result.metrics["hit_rate@5"] == pytest.approx(1.0)
    assert result.auc == pytest.approx(0.0)


def test_auc_skip_behavior_for_one_class_impressions():
    summary = aggregate_scored_candidates(
        [_candidate(0, 1, 1.0), _candidate(1, 1, 0.5)]
    )

    assert summary["metrics"]["auc"] is None
    assert summary["auc_skipped_impressions"] == {"one_class": 1}


def test_deterministic_tie_breaking_uses_original_position():
    candidates = [_candidate(0, 0, 1.0), _candidate(1, 1, 1.0)]

    assert deterministic_rank_order(candidates) == [0, 1]
    assert evaluate_scored_candidates(candidates).metrics["mrr"] == pytest.approx(0.5)


def test_no_positive_labels_are_counted_with_zero_ranking_metrics():
    result = evaluate_scored_candidates([_candidate(0, 0, 0.2), _candidate(1, 0, 0.1)])

    assert result.metrics["mrr"] == 0.0
    assert result.metrics["ndcg@5"] == 0.0
    assert result.metrics["recall@5"] == 0.0
    assert result.auc is None


def test_unlabeled_missing_and_non_finite_scores_are_skipped():
    assert evaluate_scored_candidates([_candidate(0, None, 1.0)]).skipped_reason == (
        "unlabeled_impression"
    )
    assert evaluate_scored_candidates([_candidate(0, 1, None)]).skipped_reason == (
        "missing_score"
    )
    assert evaluate_scored_candidates([_candidate(0, 1, float("nan"))]).skipped_reason == (
        "non_finite_score"
    )
