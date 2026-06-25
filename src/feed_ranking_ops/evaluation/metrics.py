from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoredCandidate:
    impression_id: str
    candidate_position: int
    click_label: int | None
    score: float | None


@dataclass(frozen=True)
class ImpressionMetricResult:
    metrics: dict[str, float]
    auc: float | None
    skipped_reason: str | None


def deterministic_rank_order(candidates: list[ScoredCandidate]) -> list[int]:
    """Return candidate indices sorted by higher score and lower source position."""
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index].score),  # type: ignore[arg-type]
            candidates[index].candidate_position,
        ),
    )


def evaluate_scored_candidates(
    candidates: list[ScoredCandidate],
) -> ImpressionMetricResult:
    if not candidates:
        return ImpressionMetricResult({}, None, "empty_impression")
    labels = [candidate.click_label for candidate in candidates]
    if any(label is None for label in labels):
        return ImpressionMetricResult({}, None, "unlabeled_impression")
    if any(candidate.score is None for candidate in candidates):
        return ImpressionMetricResult({}, None, "missing_score")
    if any(not math.isfinite(float(candidate.score)) for candidate in candidates):
        return ImpressionMetricResult({}, None, "non_finite_score")

    numeric_labels = [int(label) for label in labels if label is not None]
    positives = sum(numeric_labels)
    order = deterministic_rank_order(candidates)
    ordered_labels = [numeric_labels[index] for index in order]
    metrics = {
        "mrr": _mrr(ordered_labels),
        "ndcg@5": _ndcg(ordered_labels, 5),
        "ndcg@10": _ndcg(ordered_labels, 10),
        "recall@5": _recall(ordered_labels, positives, 5),
        "recall@10": _recall(ordered_labels, positives, 10),
        "hit_rate@5": _hit_rate(ordered_labels, 5),
        "hit_rate@10": _hit_rate(ordered_labels, 10),
    }
    auc = _auc(numeric_labels, [float(candidate.score) for candidate in candidates])
    return ImpressionMetricResult(metrics, auc, None)


def aggregate_scored_candidates(
    candidates: list[ScoredCandidate],
) -> dict[str, Any]:
    grouped: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.impression_id].append(candidate)

    metric_values: dict[str, list[float]] = defaultdict(list)
    auc_values: list[float] = []
    skipped = Counter()
    auc_skipped = Counter()

    for group in grouped.values():
        result = evaluate_scored_candidates(group)
        if result.skipped_reason is not None:
            skipped[result.skipped_reason] += 1
            continue
        for metric_name, value in result.metrics.items():
            metric_values[metric_name].append(value)
        if result.auc is None:
            auc_skipped["one_class"] += 1
        else:
            auc_values.append(result.auc)

    metrics = {
        metric_name: float(np.mean(values)) if values else None
        for metric_name, values in sorted(metric_values.items())
    }
    metrics["auc"] = float(np.mean(auc_values)) if auc_values else None
    return {
        "metrics": metrics,
        "n_impressions": len(grouped),
        "n_evaluated_impressions": max(
            [len(values) for values in metric_values.values()],
            default=0,
        ),
        "skipped_impressions": dict(sorted(skipped.items())),
        "auc_evaluated_impressions": len(auc_values),
        "auc_skipped_impressions": dict(sorted(auc_skipped.items())),
        "metric_definitions": metric_definitions(),
    }


def predicted_ranks(candidates: list[ScoredCandidate]) -> dict[int, int]:
    order = deterministic_rank_order(candidates)
    return {
        candidates[index].candidate_position: rank
        for rank, index in enumerate(order, start=1)
    }


def metric_definitions() -> dict[str, str]:
    return {
        "mrr": "Reciprocal rank of the first clicked candidate, averaged by impression.",
        "ndcg@5": "NDCG at 5 with binary click labels and deterministic score ties.",
        "ndcg@10": "NDCG at 10 with binary click labels and deterministic score ties.",
        "recall@5": "Clicked candidates in the top 5 divided by all clicked candidates.",
        "recall@10": "Clicked candidates in the top 10 divided by all clicked candidates.",
        "hit_rate@5": "1 if any clicked candidate appears in the top 5, else 0.",
        "hit_rate@10": "1 if any clicked candidate appears in the top 10, else 0.",
        "auc": "Pairwise impression-level AUC; one-class impressions are skipped.",
        "no_positive_policy": (
            "MRR, NDCG, recall, and hit-rate are scored as 0 for labeled impressions "
            "with no clicked candidates. They are counted, not hidden."
        ),
    }


def _mrr(ordered_labels: list[int]) -> float:
    for rank, label in enumerate(ordered_labels, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def _ndcg(ordered_labels: list[int], k: int) -> float:
    gains = ordered_labels[:k]
    ideal = sorted(ordered_labels, reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def _dcg(labels: list[int]) -> float:
    return sum(label / math.log2(index + 2) for index, label in enumerate(labels))


def _recall(ordered_labels: list[int], positives: int, k: int) -> float:
    if positives == 0:
        return 0.0
    return sum(ordered_labels[:k]) / positives


def _hit_rate(ordered_labels: list[int], k: int) -> float:
    return 1.0 if any(label == 1 for label in ordered_labels[:k]) else 0.0


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive_scores = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    total = len(positive_scores) * len(negative_scores)
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total
