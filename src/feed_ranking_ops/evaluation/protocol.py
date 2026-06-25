from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.baselines import (
    CategoryAffinityBaseline,
    GlobalPopularityBaseline,
    OriginalOrderBaseline,
    RankingBaseline,
    TfidfContentSimilarityBaseline,
    TimeDecayedPopularityBaseline,
    default_baseline_names,
    make_baselines,
)
from feed_ranking_ops.evaluation.metrics import (
    ScoredCandidate,
    aggregate_scored_candidates,
    predicted_ranks,
)
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    NewsItem,
    ProcessedDataset,
    load_processed_dataset,
)

DEFAULT_HALF_LIVES = [6.0, 24.0, 72.0]
SELECTION_METRIC = "ndcg@10"


def run_baseline_protocol(
    *,
    processed_dir: Path,
    reports_dir: Path,
    baseline_names: list[str] | None = None,
    half_life_hours: list[float] | None = None,
    limit_impressions: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run validation selection and final logged-candidate ranking evaluation."""
    del seed
    if limit_impressions is not None and limit_impressions <= 0:
        raise ValueError("limit_impressions must be positive when provided")
    selected_names = baseline_names or default_baseline_names()
    half_lives = half_life_hours or DEFAULT_HALF_LIVES
    dataset = load_processed_dataset(processed_dir)
    train = _limit(dataset.behaviors["train"], limit_impressions)
    validation = _limit(dataset.behaviors["validation"], limit_impressions)
    test = _limit(dataset.behaviors["test"], limit_impressions)

    validation_baselines = make_baselines(selected_names, half_life_hours=half_lives)
    validation_result = _fit_and_evaluate(
        validation_baselines,
        fit_behaviors=train,
        eval_behaviors=validation,
        news=dataset.news,
        fit_partitions=["train"],
        eval_partition="validation",
    )
    selected_baselines = _select_validation_configurations(validation_result["runs"])
    final_baselines = [_clone_baseline(baseline) for baseline in selected_baselines]
    test_result = _fit_and_evaluate(
        final_baselines,
        fit_behaviors=[*train, *validation],
        eval_behaviors=test,
        news=dataset.news,
        fit_partitions=["train", "validation"],
        eval_partition="test",
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_predictions_path = reports_dir / "validation_predictions.parquet"
    test_predictions_path = reports_dir / "test_predictions.parquet"
    _write_predictions(validation_result["predictions"], validation_predictions_path)
    _write_predictions(test_result["predictions"], test_predictions_path)

    protocol = {
        "evaluation_type": "logged_candidate_ranking",
        "candidate_scope": (
            "Ranks only the candidate articles already present in each logged MIND "
            "impression; this is not full-catalog retrieval evaluation."
        ),
        "selection_metric": SELECTION_METRIC,
        "smoke_test": limit_impressions is not None,
        "limit_impressions": limit_impressions,
        "validation_fit_partitions": ["train"],
        "validation_eval_partition": "validation",
        "test_fit_partitions": ["train", "validation"],
        "test_eval_partition": "test",
        "test_labels_used_for_selection": False,
        "half_life_grid_hours": half_lives,
        "selected_hyperparameters": _selected_hyperparameters(selected_baselines),
        "history_missing_counts": dataset.history_missing_counts,
    }
    validation_metrics = _metrics_document(
        dataset,
        validation_result,
        partition_sizes={"train": len(train), "validation": len(validation)},
        protocol=protocol,
    )
    test_metrics = _metrics_document(
        dataset,
        test_result,
        partition_sizes={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        protocol=protocol,
    )
    output_paths = {
        "validation_metrics": reports_dir / "validation_metrics.json",
        "test_metrics": reports_dir / "test_metrics.json",
        "protocol": reports_dir / "protocol.json",
        "report": reports_dir / "model_comparison.md",
        "validation_predictions": validation_predictions_path,
        "test_predictions": test_predictions_path,
    }
    _write_json(output_paths["validation_metrics"], validation_metrics)
    _write_json(output_paths["test_metrics"], test_metrics)
    _write_json(output_paths["protocol"], protocol)
    output_paths["report"].write_text(
        render_model_comparison(validation_metrics, test_metrics, protocol),
        encoding="utf-8",
    )
    return {
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "protocol": protocol,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def render_model_comparison(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    lines = [
        "# Offline Logged-Candidate Baseline Comparison",
        "",
        "This report evaluates ranking within candidate sets already present in logged "
        "MIND impressions. It does not measure full-catalog candidate retrieval quality.",
        "",
        "## Protocol",
        "",
        "- Validation: fit on chronological train, evaluate on chronological validation.",
        "- Selection: tune hyperparameters using validation metrics only.",
        "- Final test: refit selected configurations on train plus validation, evaluate once on test.",
        "- Behavior histories come only from each logged row; clicked candidates from the same row are not appended before scoring.",
        f"- Smoke-test limit: {protocol['limit_impressions'] if protocol['smoke_test'] else 'none'}",
        "",
        "## Metric Definitions",
        "",
    ]
    definitions = next(iter(validation_metrics["baselines"].values()))["metric_definitions"]
    lines.extend(f"- `{name}`: {definition}" for name, definition in definitions.items())
    lines.extend(
        [
            "",
            "## Validation Results",
            "",
            _metrics_table(validation_metrics),
            "",
            "## Selected Hyperparameters",
            "",
        ]
    )
    for name, config in protocol["selected_hyperparameters"].items():
        lines.append(f"- `{name}`: {config}")
    lines.extend(
        [
            "",
            "## Final Test Results",
            "",
            _metrics_table(test_metrics),
            "",
            "## Coverage And Skips",
            "",
            f"- History missing news references: {protocol['history_missing_counts']}",
            "- Candidate missing news references: 0; loading fails before evaluation otherwise.",
            "- AUC is skipped for one-class impressions.",
            "- Unlabeled inference-style impressions are counted as skipped, not imputed.",
            "",
            "## Interpretation",
            "",
            "Popularity baselines test whether logged clicks concentrate on a small set of "
            "articles. Category and TF-IDF baselines test whether row-supplied user history "
            "contains enough signal to rank the logged candidates. Original order is a "
            "diagnostic source-order baseline.",
            "",
            "## Limitations",
            "",
            "- Logged-candidate metrics are affected by exposure bias from the original system.",
            "- They do not show whether a model can retrieve relevant articles from the full catalog.",
            "- Click labels are implicit feedback, not editorial relevance judgments.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fit_and_evaluate(
    baselines: list[RankingBaseline],
    *,
    fit_behaviors: list[BehaviorImpression],
    eval_behaviors: list[BehaviorImpression],
    news: dict[str, NewsItem],
    fit_partitions: list[str],
    eval_partition: str,
) -> dict[str, Any]:
    runs = []
    all_predictions: list[dict[str, Any]] = []
    for baseline in baselines:
        baseline.fit(fit_behaviors, news, fitting_partitions=fit_partitions)
        predictions, score_metadata = _score_baseline(
            baseline,
            eval_behaviors,
            news,
            partition=eval_partition,
        )
        _validate_prediction_rows(predictions, eval_behaviors, run_name(baseline))
        scored = [
            ScoredCandidate(
                impression_id=row["impression_id"],
                candidate_position=row["candidate_position"],
                click_label=row["click_label"],
                score=row["score"],
            )
            for row in predictions
        ]
        metrics = aggregate_scored_candidates(scored)
        run = {
            "baseline_name": run_name(baseline),
            "baseline_family": baseline.name,
            "config": baseline.config(),
            "fit_metadata": baseline.metadata(),
            "score_metadata": dict(sorted(score_metadata.items())),
            **metrics,
        }
        runs.append({"baseline": baseline, "summary": run})
        all_predictions.extend(predictions)
    return {
        "runs": runs,
        "predictions": all_predictions,
        "partition": eval_partition,
    }


def _score_baseline(
    baseline: RankingBaseline,
    behaviors: list[BehaviorImpression],
    news: dict[str, NewsItem],
    *,
    partition: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    predictions: list[dict[str, Any]] = []
    score_metadata: Counter[str] = Counter()
    for behavior in behaviors:
        result = baseline.score(behavior, news)
        score_metadata.update(result.metadata)
        if len(result.scores) != len(behavior.candidates):
            raise ValueError(
                f"{run_name(baseline)} produced {len(result.scores)} scores for "
                f"{len(behavior.candidates)} candidates"
            )
        scored = [
            ScoredCandidate(
                impression_id=behavior.impression_id,
                candidate_position=candidate.position,
                click_label=candidate.clicked,
                score=score,
            )
            for candidate, score in zip(behavior.candidates, result.scores, strict=True)
        ]
        ranks = predicted_ranks(scored)
        for candidate, score in zip(behavior.candidates, result.scores, strict=True):
            predictions.append(
                {
                    "partition": partition,
                    "impression_id": behavior.impression_id,
                    "user_id": behavior.user_id,
                    "timestamp": behavior.timestamp,
                    "candidate_news_id": candidate.news_id,
                    "candidate_position": candidate.position,
                    "click_label": candidate.clicked,
                    "baseline_name": run_name(baseline),
                    "baseline_family": baseline.name,
                    "score": float(score),
                    "predicted_rank": ranks[candidate.position],
                }
            )
    return predictions, score_metadata


def _validate_prediction_rows(
    predictions: list[dict[str, Any]],
    behaviors: list[BehaviorImpression],
    baseline_name: str,
) -> None:
    expected: dict[tuple[str, int], tuple[str, int | None]] = {
        (behavior.impression_id, candidate.position): (candidate.news_id, candidate.clicked)
        for behavior in behaviors
        for candidate in behavior.candidates
    }
    seen: set[tuple[str, int]] = set()
    for row in predictions:
        if row["baseline_name"] != baseline_name:
            raise ValueError("prediction row crossed baseline groups")
        key = (row["impression_id"], row["candidate_position"])
        if key in seen:
            raise ValueError("duplicate candidate prediction for a baseline/impression position")
        seen.add(key)
        if key not in expected:
            raise ValueError("prediction row crossed impression groups")
        news_id, label = expected[key]
        if row["candidate_news_id"] != news_id or row["click_label"] != label:
            raise ValueError("original labels or candidate ordering were not preserved")
    if seen != set(expected):
        raise ValueError("missing candidate predictions for baseline")


def _select_validation_configurations(
    runs: list[dict[str, Any]],
) -> list[RankingBaseline]:
    selected: list[RankingBaseline] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_family.setdefault(run["summary"]["baseline_family"], []).append(run)
    for family, family_runs in sorted(by_family.items()):
        if family == "time_decayed_popularity":
            selected.append(_best_validation_run(family_runs)["baseline"])
        else:
            selected.append(family_runs[0]["baseline"])
    return selected


def _best_validation_run(family_runs: list[dict[str, Any]]) -> dict[str, Any]:
    def key(run: dict[str, Any]) -> tuple[float, float, float]:
        metrics = run["summary"]["metrics"]
        ndcg = metrics.get(SELECTION_METRIC)
        mrr = metrics.get("mrr")
        half_life = run["summary"]["config"].get("half_life_hours", float("inf"))
        return (
            float(ndcg) if ndcg is not None else float("-inf"),
            float(mrr) if mrr is not None else float("-inf"),
            -float(half_life),
        )

    return max(family_runs, key=key)


def _selected_hyperparameters(baselines: list[RankingBaseline]) -> dict[str, dict[str, Any]]:
    return {run_name(baseline): baseline.config() for baseline in baselines}


def _clone_baseline(baseline: RankingBaseline) -> RankingBaseline:
    if isinstance(baseline, OriginalOrderBaseline):
        return OriginalOrderBaseline()
    if isinstance(baseline, GlobalPopularityBaseline):
        return GlobalPopularityBaseline(fallback_score=baseline.fallback_score)
    if isinstance(baseline, TimeDecayedPopularityBaseline):
        return TimeDecayedPopularityBaseline(
            half_life_hours=baseline.half_life_hours,
            fallback_score=baseline.fallback_score,
        )
    if isinstance(baseline, CategoryAffinityBaseline):
        return CategoryAffinityBaseline(
            category_weight=baseline.category_weight,
            subcategory_weight=baseline.subcategory_weight,
            fallback_score=baseline.fallback_score,
        )
    if isinstance(baseline, TfidfContentSimilarityBaseline):
        return TfidfContentSimilarityBaseline(fallback_score=baseline.fallback_score)
    raise TypeError(f"Cannot clone baseline {type(baseline).__name__}")


def _metrics_document(
    dataset: ProcessedDataset,
    result: dict[str, Any],
    *,
    partition_sizes: dict[str, int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    del dataset
    return {
        "partition": result["partition"],
        "partition_sizes": partition_sizes,
        "smoke_test": protocol["smoke_test"],
        "limit_impressions": protocol["limit_impressions"],
        "baselines": {
            run["summary"]["baseline_name"]: {
                key: value
                for key, value in run["summary"].items()
                if key != "baseline_name"
            }
            for run in result["runs"]
        },
    }


def _metrics_table(metrics_document: dict[str, Any]) -> str:
    lines = [
        "| Baseline | MRR | NDCG@5 | NDCG@10 | Recall@5 | Hit@5 | AUC | Evaluated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in metrics_document["baselines"].items():
        metrics = summary["metrics"]
        lines.append(
            f"| {name} | {_format_metric(metrics.get('mrr'))} | "
            f"{_format_metric(metrics.get('ndcg@5'))} | "
            f"{_format_metric(metrics.get('ndcg@10'))} | "
            f"{_format_metric(metrics.get('recall@5'))} | "
            f"{_format_metric(metrics.get('hit_rate@5'))} | "
            f"{_format_metric(metrics.get('auc'))} | "
            f"{summary['n_evaluated_impressions']} |"
        )
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _limit(
    behaviors: list[BehaviorImpression],
    limit_impressions: int | None,
) -> list[BehaviorImpression]:
    return behaviors[:limit_impressions] if limit_impressions is not None else behaviors


def _write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("partition", pa.string()),
            pa.field("impression_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("candidate_news_id", pa.string()),
            pa.field("candidate_position", pa.int32()),
            pa.field("click_label", pa.int8()),
            pa.field("baseline_name", pa.string()),
            pa.field("baseline_family", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("predicted_rank", pa.int32()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run_name(baseline: RankingBaseline) -> str:
    return getattr(baseline, "selection_name", baseline.name)
