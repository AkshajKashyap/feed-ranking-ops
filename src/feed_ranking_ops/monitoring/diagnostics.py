from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from feed_ranking_ops.evaluation.metrics import (
    ImpressionMetricResult,
    ScoredCandidate,
    evaluate_scored_candidates,
)
from feed_ranking_ops.evaluation.processed import BehaviorImpression
from feed_ranking_ops.serving.policy import PolicyRuntime


@dataclass
class MetricAccumulator:
    n_impressions: int = 0
    n_evaluated: int = 0
    metric_values: dict[str, list[float]] = field(default_factory=dict)
    auc_values: list[float] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)

    def add(self, result: ImpressionMetricResult) -> None:
        self.n_impressions += 1
        if result.skipped_reason:
            self.skipped[result.skipped_reason] += 1
            return
        self.n_evaluated += 1
        for name, value in result.metrics.items():
            self.metric_values.setdefault(name, []).append(value)
        if result.auc is not None:
            self.auc_values.append(result.auc)

    def summary(self) -> dict[str, Any]:
        metrics = {
            name: float(np.mean(values)) if values else None
            for name, values in sorted(self.metric_values.items())
        }
        metrics["auc"] = (
            float(np.mean(self.auc_values)) if self.auc_values else None
        )
        return {
            "n_impressions": self.n_impressions,
            "n_evaluated_impressions": self.n_evaluated,
            "metrics": metrics,
            "skipped_impressions": dict(sorted(self.skipped.items())),
        }


def compute_policy_slice_diagnostics(
    behaviors: list[BehaviorImpression],
    runtime: PolicyRuntime,
) -> dict[str, Any]:
    history_slices: dict[str, MetricAccumulator] = {}
    candidate_slices: dict[str, MetricAccumulator] = {}
    empty_slices: dict[str, MetricAccumulator] = {}
    category_slices: dict[str, MetricAccumulator] = {}
    all_metrics = MetricAccumulator()
    candidate_total = 0
    known_category = 0
    known_subcategory = 0
    impressions_with_ties = 0
    top10_concentrations: list[float] = []
    missing_candidate_count = 0

    for behavior in behaviors:
        candidate_ids = [candidate.news_id for candidate in behavior.candidates]
        score_result = runtime.score_candidates(
            history_news_ids=behavior.history_news_ids,
            candidate_news_ids=candidate_ids,
        )
        score_by_index = {
            candidate.original_position: candidate.score
            for candidate in score_result.candidates
        }
        scored = [
            ScoredCandidate(
                impression_id=behavior.impression_id,
                candidate_position=candidate.position,
                click_label=candidate.clicked,
                score=score_by_index.get(candidate_index),
            )
            for candidate_index, candidate in enumerate(behavior.candidates)
        ]
        result = evaluate_scored_candidates(scored)
        all_metrics.add(result)
        _accumulator(
            history_slices,
            history_length_bucket(len(behavior.history_news_ids)),
        ).add(result)
        _accumulator(
            candidate_slices,
            candidate_count_bucket(len(behavior.candidates)),
        ).add(result)
        _accumulator(
            empty_slices,
            "empty" if not behavior.history_news_ids else "non_empty",
        ).add(result)

        ordered_scores = sorted(
            score_result.candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.original_position,
            ),
        )
        if ordered_scores:
            top_item = runtime.news.get(ordered_scores[0].news_id)
            top_category = (
                top_item.category if top_item and top_item.category else "<missing>"
            )
            _accumulator(category_slices, top_category).add(result)
            top_categories = [
                (
                    runtime.news[candidate.news_id].category
                    if runtime.news[candidate.news_id].category
                    else "<missing>"
                )
                for candidate in ordered_scores[:10]
            ]
            counts = Counter(top_categories)
            top10_concentrations.append(
                max(counts.values()) / len(top_categories)
            )
        scores = [candidate.score for candidate in score_result.candidates]
        impressions_with_ties += int(len(scores) != len(set(scores)))
        missing_candidate_count += len(score_result.missing_candidate_ids)
        for candidate in behavior.candidates:
            candidate_total += 1
            item = runtime.news.get(candidate.news_id)
            known_category += int(item is not None and bool(item.category))
            known_subcategory += int(item is not None and bool(item.subcategory))

    impression_count = len(behaviors)
    return {
        "overall": all_metrics.summary(),
        "by_history_length": _slice_summaries(history_slices),
        "by_candidate_count": _slice_summaries(candidate_slices),
        "by_empty_history": _slice_summaries(empty_slices),
        "by_top_ranked_category": _slice_summaries(category_slices),
        "coverage": {
            "candidate_count": candidate_total,
            "known_category_fraction": _rate(known_category, candidate_total),
            "known_subcategory_fraction": _rate(
                known_subcategory,
                candidate_total,
            ),
            "missing_candidate_fraction": _rate(
                missing_candidate_count,
                candidate_total,
            ),
        },
        "score_ties": {
            "impressions_with_ties": impressions_with_ties,
            "fraction": _rate(impressions_with_ties, impression_count),
        },
        "average_top10_category_concentration": (
            float(np.mean(top10_concentrations))
            if top10_concentrations
            else None
        ),
    }


def build_input_profile(
    behaviors: list[BehaviorImpression],
    runtime: PolicyRuntime,
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    history_buckets: Counter[str] = Counter()
    candidate_buckets: Counter[str] = Counter()
    empty_histories = 0
    candidate_total = 0
    missing_candidates = 0
    for behavior in behaviors:
        history_buckets[history_length_bucket(len(behavior.history_news_ids))] += 1
        candidate_buckets[candidate_count_bucket(len(behavior.candidates))] += 1
        empty_histories += int(not behavior.history_news_ids)
        for candidate in behavior.candidates:
            candidate_total += 1
            item = runtime.news.get(candidate.news_id)
            if item is None:
                missing_candidates += 1
            else:
                category_counts[item.category or "<missing>"] += 1
    return {
        "n_impressions": len(behaviors),
        "n_candidates": candidate_total,
        "candidate_category_distribution": normalize_distribution(
            category_counts
        ),
        "history_length_distribution": normalize_distribution(history_buckets),
        "candidate_count_distribution": normalize_distribution(
            candidate_buckets
        ),
        "empty_history_rate": _rate(empty_histories, len(behaviors)),
        "missing_candidate_rate": _rate(
            missing_candidates,
            candidate_total,
        ),
    }


def compare_profiles(
    reference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_category_js_divergence": jensen_shannon_divergence(
            reference["candidate_category_distribution"],
            comparison["candidate_category_distribution"],
        ),
        "history_length_js_divergence": jensen_shannon_divergence(
            reference["history_length_distribution"],
            comparison["history_length_distribution"],
        ),
        "candidate_count_js_divergence": jensen_shannon_divergence(
            reference["candidate_count_distribution"],
            comparison["candidate_count_distribution"],
        ),
        "empty_history_rate_absolute_difference": abs(
            float(reference["empty_history_rate"])
            - float(comparison["empty_history_rate"])
        ),
        "missing_candidate_rate_absolute_difference": abs(
            float(reference["missing_candidate_rate"])
            - float(comparison["missing_candidate_rate"])
        ),
    }


def jensen_shannon_divergence(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    keys = sorted(set(first).union(second))
    if not keys:
        return 0.0
    first_total = sum(max(float(first.get(key, 0.0)), 0.0) for key in keys)
    second_total = sum(max(float(second.get(key, 0.0)), 0.0) for key in keys)
    if first_total == 0.0 and second_total == 0.0:
        return 0.0
    first_values = [
        max(float(first.get(key, 0.0)), 0.0) / first_total
        if first_total
        else 0.0
        for key in keys
    ]
    second_values = [
        max(float(second.get(key, 0.0)), 0.0) / second_total
        if second_total
        else 0.0
        for key in keys
    ]
    midpoint = [
        (first_value + second_value) / 2.0
        for first_value, second_value in zip(
            first_values,
            second_values,
            strict=True,
        )
    ]
    divergence = 0.5 * _kl_divergence(first_values, midpoint)
    divergence += 0.5 * _kl_divergence(second_values, midpoint)
    return float(divergence)


def normalize_distribution(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        key: value / total
        for key, value in sorted(counts.items())
    } if total else {}


def history_length_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 20:
        return "6-20"
    return "21+"


def candidate_count_bucket(value: int) -> str:
    if value <= 10:
        return "1-10"
    if value <= 50:
        return "11-50"
    return "51+"


def _kl_divergence(values: list[float], midpoint: list[float]) -> float:
    return sum(
        value * math.log2(value / middle)
        for value, middle in zip(values, midpoint, strict=True)
        if value > 0.0 and middle > 0.0
    )


def _accumulator(
    values: dict[str, MetricAccumulator],
    key: str,
) -> MetricAccumulator:
    if key not in values:
        values[key] = MetricAccumulator()
    return values[key]


def _slice_summaries(
    values: dict[str, MetricAccumulator],
) -> dict[str, Any]:
    return {
        key: accumulator.summary()
        for key, accumulator in sorted(values.items())
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
