from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any


def agreement_metrics(
    reference_ids: list[str],
    candidate_ids: list[str],
    *,
    cutoffs: Iterable[int] = (10, 20, 50, 100),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "top1_agreement": _top1_agreement(reference_ids, candidate_ids),
        "ordered_list_agreement": 1.0 if reference_ids == candidate_ids else 0.0,
        "first_differing_rank": _first_differing_rank(reference_ids, candidate_ids),
    }
    for cutoff in cutoffs:
        ref = reference_ids[:cutoff]
        cand = candidate_ids[:cutoff]
        ref_set = set(ref)
        cand_set = set(cand)
        overlap = len(ref_set & cand_set)
        union = len(ref_set | cand_set)
        metrics[f"overlap_count@{cutoff}"] = overlap
        metrics[f"set_recall@{cutoff}"] = overlap / len(ref_set) if ref_set else None
        metrics[f"jaccard@{cutoff}"] = overlap / union if union else None
        metrics[f"mean_rank_displacement@{cutoff}"] = _mean_rank_displacement(ref, cand)
    return metrics


def aggregate_agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), int | float)})
    return {
        key: float(mean(float(row[key]) for row in rows if row.get(key) is not None))
        for key in keys
        if any(row.get(key) is not None for row in rows)
    }


def representation_loss_metrics(
    sparse_metrics: dict[str, float | None],
    dense_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    keys = sorted(set(sparse_metrics) | set(dense_metrics))
    output: dict[str, float | None] = {}
    for key in keys:
        sparse_value = sparse_metrics.get(key)
        dense_value = dense_metrics.get(key)
        output[f"dense_minus_sparse_{key}"] = (
            None if sparse_value is None or dense_value is None else dense_value - sparse_value
        )
    return output


def _top1_agreement(reference_ids: list[str], candidate_ids: list[str]) -> float:
    if not reference_ids and not candidate_ids:
        return 1.0
    if not reference_ids or not candidate_ids:
        return 0.0
    return 1.0 if reference_ids[0] == candidate_ids[0] else 0.0


def _first_differing_rank(reference_ids: list[str], candidate_ids: list[str]) -> int | None:
    max_len = max(len(reference_ids), len(candidate_ids))
    for index in range(max_len):
        left = reference_ids[index] if index < len(reference_ids) else None
        right = candidate_ids[index] if index < len(candidate_ids) else None
        if left != right:
            return index + 1
    return None


def _mean_rank_displacement(reference_ids: list[str], candidate_ids: list[str]) -> float | None:
    ref_rank = {news_id: index for index, news_id in enumerate(reference_ids, start=1)}
    displacements = [
        abs(index - ref_rank[news_id])
        for index, news_id in enumerate(candidate_ids, start=1)
        if news_id in ref_rank
    ]
    return float(mean(displacements)) if displacements else None
