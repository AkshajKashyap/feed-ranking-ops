from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from feed_ranking_ops.retrieval.exact import RetrievalResult

RECALL_CUTOFFS = [10, 20, 50, 100]
HIT_CUTOFFS = [10, 20, 50, 100]
NDCG_CUTOFFS = [10, 20, 100]


def evaluate_retrieval_results(
    results: list[RetrievalResult],
    *,
    catalog_size_total: int,
) -> dict[str, Any]:
    valid_results = [
        result for result in results if result.available_target_news_ids
    ]
    skipped = {
        "no_clicked_target": sum(1 for result in results if not result.target_news_ids),
        "missing_or_unavailable_targets": sum(
            1 for result in results if result.target_news_ids and not result.available_target_news_ids
        ),
    }
    metrics: dict[str, float | None] = {}
    for cutoff in RECALL_CUTOFFS:
        metrics[f"recall@{cutoff}"] = _average(
            _recall_at(result, cutoff) for result in valid_results
        )
    for cutoff in HIT_CUTOFFS:
        metrics[f"hit_rate@{cutoff}"] = _average(
            _hit_at(result, cutoff) for result in valid_results
        )
    metrics["mrr@100"] = _average(_mrr_at(result, 100) for result in valid_results)
    for cutoff in NDCG_CUTOFFS:
        metrics[f"ndcg@{cutoff}"] = _average(
            _ndcg_at(result, cutoff) for result in valid_results
        )

    available_targets = sum(len(result.available_target_news_ids) for result in results)
    metadata_targets = sum(
        len(result.available_target_news_ids) + result.unavailable_target_count
        for result in results
    )
    retrieved_ids = {
        item.news_id for result in valid_results for item in result.retrieved
    }
    retrieved_history = sum(
        1 for result in valid_results for item in result.retrieved if item.was_in_history
    )
    retrieved_count = sum(len(result.retrieved) for result in valid_results)
    catalog_sizes = [result.catalog_size for result in valid_results]
    latencies = [result.latency_seconds for result in valid_results]
    return {
        "metrics": metrics,
        "n_queries": len(results),
        "n_valid_queries": len(valid_results),
        "skipped_queries": skipped,
        "mean_catalog_size": _safe_mean(catalog_sizes),
        "median_catalog_size": _safe_median(catalog_sizes),
        "clicked_target_availability_rate": (
            available_targets / metadata_targets if metadata_targets else None
        ),
        "fallback_query_rate": (
            sum(1 for result in valid_results if result.fallback_used) / len(valid_results)
            if valid_results
            else None
        ),
        "retrieved_history_article_rate": (
            retrieved_history / retrieved_count if retrieved_count else None
        ),
        "unique_recommendation_coverage": len(retrieved_ids),
        "catalog_coverage": (
            len(retrieved_ids) / catalog_size_total if catalog_size_total else None
        ),
        "average_known_history_items": _safe_mean(
            [result.known_history_count for result in valid_results]
        ),
        "unknown_history_reference_count": sum(
            result.unknown_history_count for result in valid_results
        ),
        "missing_clicked_target_metadata_count": sum(
            result.missing_target_metadata_count for result in results
        ),
        "efficiency": {
            "total_query_count": len(valid_results),
            "mean_candidates_scored_per_query": _safe_mean(catalog_sizes),
            "total_scoring_time_seconds": _round_time(sum(latencies)),
            "mean_latency_seconds": _round_optional_time(_safe_mean(latencies)),
            "p50_latency_seconds": _round_optional_time(_percentile(latencies, 50)),
            "p95_latency_seconds": _round_optional_time(_percentile(latencies, 95)),
            "p99_latency_seconds": _round_optional_time(_percentile(latencies, 99)),
            "timing_precision_seconds": 0.01,
        },
        "metric_definitions": metric_definitions(),
    }


def query_summary(result: RetrievalResult) -> dict[str, Any]:
    return {
        "partition": result.query.partition,
        "impression_id": result.query.impression_id,
        "target_article_ids": list(result.target_news_ids),
        "retrieved_target_count@10": _retrieved_target_count(result, 10),
        "retrieved_target_count@20": _retrieved_target_count(result, 20),
        "retrieved_target_count@50": _retrieved_target_count(result, 50),
        "retrieved_target_count@100": _retrieved_target_count(result, 100),
        "reciprocal_rank@100": _mrr_at(result, 100),
        "catalog_size": result.catalog_size,
        "known_history_count": result.known_history_count,
        "unknown_history_count": result.unknown_history_count,
        "unavailable_target_count": result.unavailable_target_count,
        "fallback_reason": result.fallback_reason,
    }


def metric_definitions() -> dict[str, str]:
    return {
        "recall@k": "Available clicked targets retrieved in the top K divided by available clicked targets.",
        "hit_rate@k": "1 if any available clicked target is retrieved in the top K, else 0.",
        "mrr@100": "Reciprocal rank of the first available clicked target up to rank 100.",
        "ndcg@k": "Binary gain NDCG for multiple clicked targets up to rank K.",
    }


def _retrieved_ids_at(result: RetrievalResult, cutoff: int) -> list[str]:
    return [item.news_id for item in result.retrieved if item.rank <= cutoff]


def _target_set(result: RetrievalResult) -> set[str]:
    return set(result.available_target_news_ids)


def _retrieved_target_count(result: RetrievalResult, cutoff: int) -> int:
    targets = _target_set(result)
    return sum(1 for news_id in _retrieved_ids_at(result, cutoff) if news_id in targets)


def _recall_at(result: RetrievalResult, cutoff: int) -> float:
    targets = _target_set(result)
    return _retrieved_target_count(result, cutoff) / len(targets) if targets else 0.0


def _hit_at(result: RetrievalResult, cutoff: int) -> float:
    return 1.0 if _retrieved_target_count(result, cutoff) > 0 else 0.0


def _mrr_at(result: RetrievalResult, cutoff: int) -> float:
    targets = _target_set(result)
    for item in result.retrieved:
        if item.rank > cutoff:
            break
        if item.news_id in targets:
            return 1.0 / item.rank
    return 0.0


def _ndcg_at(result: RetrievalResult, cutoff: int) -> float:
    targets = _target_set(result)
    gains = [
        1 if item.news_id in targets else 0
        for item in result.retrieved
        if item.rank <= cutoff
    ]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_hits = min(len(targets), cutoff)
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def _average(values) -> float | None:
    values = list(values)
    return float(mean(values)) if values else None


def _safe_mean(values: list[float | int]) -> float | None:
    return float(mean(values)) if values else None


def _safe_median(values: list[float | int]) -> float | None:
    return float(median(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((percentile / 100) * len(ordered)) - 1),
    )
    return float(ordered[index])


def _round_time(value: float) -> float:
    return round(float(value), 2)


def _round_optional_time(value: float | None) -> float | None:
    return None if value is None else _round_time(value)
