from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.metrics import (
    ScoredCandidate,
    evaluate_scored_candidates,
    metric_definitions,
    predicted_ranks,
)
from feed_ranking_ops.evaluation.processed import load_processed_dataset
from feed_ranking_ops.ranking.features import (
    CandidateFeatureBuilder,
    CandidateFeatureMatrix,
)
from feed_ranking_ops.ranking.models import (
    RankerConfig,
    build_ranker,
    default_ranker_configs,
    fit_ranker,
    model_diagnostics,
    predict_ranker_scores,
)

SELECTION_METRIC = "ndcg@10"
BASELINE_NAMES = [
    "original_order",
    "global_popularity",
    "time_decayed_popularity",
    "category_affinity",
    "tfidf_content_similarity",
]


def run_ltr_protocol(
    *,
    processed_dir: Path,
    reports_dir: Path,
    limit_impressions: int | None = None,
    seed: int = 42,
    ranker_configs: list[RankerConfig] | None = None,
) -> dict[str, Any]:
    if limit_impressions is not None and limit_impressions <= 0:
        raise ValueError("limit_impressions must be positive when provided")
    configs = ranker_configs or default_ranker_configs()
    if not configs:
        raise ValueError("At least one ranker configuration is required")

    total_start = perf_counter()
    stage_timings: dict[str, float] = {}
    load_start = perf_counter()
    dataset = load_processed_dataset(
        processed_dir,
        limit_impressions=limit_impressions,
    )
    train = dataset.behaviors["train"]
    validation = dataset.behaviors["validation"]
    test = dataset.behaviors["test"]
    stage_timings["data_loading_seconds"] = perf_counter() - load_start
    if not train or not validation or not test:
        raise ValueError("Train, validation, and test partitions must all be non-empty")

    validation_features_start = perf_counter()
    validation_builder = CandidateFeatureBuilder().fit(
        train,
        dataset.news,
        fitting_partitions=["train"],
    )
    train_matrix = validation_builder.transform(
        train,
        dataset.news,
        chronological_training=True,
    )
    validation_matrix = validation_builder.transform(validation, dataset.news)
    stage_timings["validation_feature_building_seconds"] = (
        perf_counter() - validation_features_start
    )

    training_start = perf_counter()
    validation_runs: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_validation_scores: np.ndarray | None = None
    logistic_diagnostics: list[dict[str, float | str]] = []
    best_logistic: dict[str, Any] | None = None
    for config in configs:
        model = fit_ranker(
            build_ranker(config, seed=seed),
            train_matrix.features,
            train_matrix.labels,
        )
        scores = predict_ranker_scores(model, validation_matrix.features)
        summary = {
            "model_name": config.name,
            "model_family": config.family,
            "config": config.parameters,
            "fit_partitions": ["train"],
            **evaluate_matrix_scores(validation_matrix, scores),
        }
        validation_runs.append(summary)
        if config.family == "logistic_regression" and (
            best_logistic is None or _is_better(summary, best_logistic)
        ):
            best_logistic = summary
            logistic_diagnostics = model_diagnostics(
                model,
                feature_names=train_matrix.feature_names,
            )
        if best is None or _is_better(summary, best):
            best = summary
            best_validation_scores = scores.copy()
    stage_timings["validation_model_training_scoring_seconds"] = (
        perf_counter() - training_start
    )
    assert best is not None
    assert best_validation_scores is not None
    selected_config = next(config for config in configs if config.name == best["model_name"])
    feature_names = list(train_matrix.feature_names)
    del train_matrix
    del validation_builder

    final_features_start = perf_counter()
    final_fit_behaviors = [*train, *validation]
    final_builder = CandidateFeatureBuilder().fit(
        final_fit_behaviors,
        dataset.news,
        fitting_partitions=["train", "validation"],
    )
    final_train_matrix = final_builder.transform(
        final_fit_behaviors,
        dataset.news,
        chronological_training=True,
    )
    test_matrix = final_builder.transform(test, dataset.news)
    stage_timings["final_feature_building_seconds"] = (
        perf_counter() - final_features_start
    )

    final_model_start = perf_counter()
    final_model = fit_ranker(
        build_ranker(selected_config, seed=seed),
        final_train_matrix.features,
        final_train_matrix.labels,
    )
    test_scores = predict_ranker_scores(final_model, test_matrix.features)
    final_diagnostics = model_diagnostics(
        final_model,
        feature_names=feature_names,
    )
    del final_train_matrix
    del final_builder
    stage_timings["final_model_training_scoring_seconds"] = (
        perf_counter() - final_model_start
    )

    baseline_start = perf_counter()
    validation_baselines = _baseline_summaries(validation_matrix)
    test_baselines = _baseline_summaries(test_matrix)
    stage_timings["baseline_evaluation_seconds"] = perf_counter() - baseline_start

    protocol = {
        "evaluation_type": "pointwise_logged_candidate_learning_to_rank",
        "candidate_scope": (
            "Ranks only candidates already present in each logged MIND impression."
        ),
        "selection_metric": SELECTION_METRIC,
        "selection_tiebreaker": "mrr_then_model_name",
        "seed": seed,
        "smoke_test": limit_impressions is not None,
        "limit_impressions": limit_impressions,
        "validation_fit_partitions": ["train"],
        "validation_eval_partition": "validation",
        "test_fit_partitions": ["train", "validation"],
        "test_eval_partition": "test",
        "validation_labels_used_for_model_fitting": False,
        "test_labels_used_for_selection": False,
        "current_impression_labels_used_as_features": False,
        "training_popularity_policy": (
            "Chronological prior-impression clicks only; the current row is scored "
            "before its labels update popularity state."
        ),
        "selected_model": {
            "model_name": selected_config.name,
            "model_family": selected_config.family,
            "config": selected_config.parameters,
        },
        "feature_names": feature_names,
        "feature_metadata": {
            "validation": validation_matrix.metadata,
            "test": test_matrix.metadata,
        },
        "partition_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "internal_holdout": dataset.split_metadata.get(
            "final_partition_type",
            "internal chronological holdout",
        ),
        "history_missing_counts": dataset.history_missing_counts,
    }
    validation_metrics = {
        "partition": "validation",
        "partition_sizes": {"train": len(train), "validation": len(validation)},
        "selected_model_name": selected_config.name,
        "rankers": {run["model_name"]: _without(run, "model_name") for run in validation_runs},
        "baselines": validation_baselines,
    }
    test_summary = {
        "model_name": selected_config.name,
        "model_family": selected_config.family,
        "config": selected_config.parameters,
        "fit_partitions": ["train", "validation"],
        **evaluate_matrix_scores(test_matrix, test_scores),
    }
    test_metrics = {
        "partition": "test",
        "partition_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "selected_model_name": selected_config.name,
        "rankers": {selected_config.name: _without(test_summary, "model_name")},
        "baselines": test_baselines,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_start = perf_counter()
    outputs = {
        "validation_metrics": reports_dir / "validation_metrics.json",
        "test_metrics": reports_dir / "test_metrics.json",
        "protocol": reports_dir / "protocol.json",
        "report": reports_dir / "model_comparison.md",
        "feature_importance": reports_dir / "feature_importance.md",
        "validation_predictions": reports_dir / "validation_predictions.parquet",
        "test_predictions": reports_dir / "test_predictions.parquet",
    }
    _write_predictions(
        validation_matrix,
        best_validation_scores,
        outputs["validation_predictions"],
        model_name=selected_config.name,
        model_family=selected_config.family,
    )
    _write_predictions(
        test_matrix,
        test_scores,
        outputs["test_predictions"],
        model_name=selected_config.name,
        model_family=selected_config.family,
    )
    stage_timings["report_writing_seconds"] = perf_counter() - write_start
    stage_timings["total_runtime_seconds"] = perf_counter() - total_start
    protocol["timing"] = stage_timings
    protocol["peak_memory_mib"] = _peak_memory_mib()
    _write_json(outputs["validation_metrics"], validation_metrics)
    _write_json(outputs["test_metrics"], test_metrics)
    _write_json(outputs["protocol"], protocol)
    outputs["report"].write_text(
        render_model_comparison(validation_metrics, test_metrics, protocol),
        encoding="utf-8",
    )
    outputs["feature_importance"].write_text(
        render_feature_importance(
            selected_config,
            validation_diagnostics=logistic_diagnostics,
            final_diagnostics=final_diagnostics,
        ),
        encoding="utf-8",
    )
    return {
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "protocol": protocol,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def evaluate_matrix_scores(
    matrix: CandidateFeatureMatrix,
    scores: np.ndarray,
) -> dict[str, Any]:
    if len(scores) != matrix.n_candidates:
        raise ValueError("Score count does not match candidate feature rows")
    metric_values: dict[str, list[float]] = {}
    auc_values: list[float] = []
    skipped: dict[str, int] = {}
    auc_one_class = 0
    for behavior_index, behavior in enumerate(matrix.behaviors):
        start = int(matrix.offsets[behavior_index])
        end = int(matrix.offsets[behavior_index + 1])
        candidates = [
            ScoredCandidate(
                impression_id=behavior.impression_id,
                candidate_position=candidate.position,
                click_label=candidate.clicked,
                score=float(score),
            )
            for candidate, score in zip(
                behavior.candidates,
                scores[start:end],
                strict=True,
            )
        ]
        result = evaluate_scored_candidates(candidates)
        if result.skipped_reason is not None:
            skipped[result.skipped_reason] = skipped.get(result.skipped_reason, 0) + 1
            continue
        for name, value in result.metrics.items():
            metric_values.setdefault(name, []).append(value)
        if result.auc is None:
            auc_one_class += 1
        else:
            auc_values.append(result.auc)
    metrics = {
        name: float(np.mean(values)) if values else None
        for name, values in sorted(metric_values.items())
    }
    metrics["auc"] = float(np.mean(auc_values)) if auc_values else None
    return {
        "metrics": metrics,
        "n_impressions": len(matrix.behaviors),
        "n_evaluated_impressions": max(
            (len(values) for values in metric_values.values()),
            default=0,
        ),
        "skipped_impressions": dict(sorted(skipped.items())),
        "auc_evaluated_impressions": len(auc_values),
        "auc_skipped_impressions": (
            {"one_class": auc_one_class} if auc_one_class else {}
        ),
        "metric_definitions": metric_definitions(),
    }


def render_model_comparison(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    selected_name = protocol["selected_model"]["model_name"]
    validation_selected = validation_metrics["rankers"][selected_name]["metrics"]
    test_selected = test_metrics["rankers"][selected_name]["metrics"]
    validation_baseline_name, validation_baseline = _best_baseline(
        validation_metrics,
        SELECTION_METRIC,
    )
    test_baseline_name, test_baseline = _best_baseline(test_metrics, SELECTION_METRIC)
    test_delta = float(test_selected[SELECTION_METRIC]) - float(
        test_baseline["metrics"][SELECTION_METRIC]
    )
    outcome = "higher" if test_delta > 0 else "lower"
    return "\n".join(
        [
            "# Logged-Candidate Learning-to-Rank Comparison",
            "",
            "This pointwise experiment ranks only candidates present in logged MIND "
            "impressions. It is not full-catalog retrieval or an official MIND test result.",
            "",
            "## Protocol",
            "",
            "- Train features use chronological prior-impression click counts.",
            "- Validation labels select the configuration by NDCG@10, then MRR.",
            "- The selected configuration is refit on train plus validation.",
            "- Internal-test labels are used exactly once for final evaluation.",
            "- No current-impression label is included in a feature.",
            f"- Smoke-test limit: {protocol['limit_impressions'] if protocol['smoke_test'] else 'none'}",
            "",
            "## Validation Results",
            "",
            _comparison_table(validation_metrics),
            "",
            "## Selected Configuration",
            "",
            f"`{protocol['selected_model']['model_name']}`: "
            f"{protocol['selected_model']['config']}",
            "",
            "## Internal Test Results",
            "",
            _comparison_table(test_metrics),
            "",
            "## Outcome",
            "",
            f"- Validation `{SELECTION_METRIC}`: selected ranker "
            f"{float(validation_selected[SELECTION_METRIC]):.4f}; strongest baseline "
            f"`{validation_baseline_name}` "
            f"{float(validation_baseline['metrics'][SELECTION_METRIC]):.4f}.",
            f"- Internal-test `{SELECTION_METRIC}`: selected ranker "
            f"{float(test_selected[SELECTION_METRIC]):.4f}; strongest baseline "
            f"`{test_baseline_name}` "
            f"{float(test_baseline['metrics'][SELECTION_METRIC]):.4f}.",
            f"- The selected ranker was {abs(test_delta):.4f} {outcome} than the strongest "
            "internal-test baseline on the selection metric. This is reported after "
            "validation selection and was not used to choose the model.",
            "",
            "## Limitations",
            "",
            "- The final partition is an internal chronological holdout, not the "
            "official MIND validation benchmark.",
            "- Pointwise click classification optimizes a surrogate objective rather "
            "than NDCG directly.",
            "- Logged clicks reflect exposure and position bias.",
            "- Feature diagnostics are associative and must not be read causally.",
        ]
    ) + "\n"


def _best_baseline(
    document: dict[str, Any],
    metric_name: str,
) -> tuple[str, dict[str, Any]]:
    return max(
        document["baselines"].items(),
        key=lambda item: (
            _metric(item[1], metric_name),
            _metric(item[1], "mrr"),
            item[0],
        ),
    )


def render_feature_importance(
    selected_config: RankerConfig,
    *,
    validation_diagnostics: list[dict[str, float | str]],
    final_diagnostics: list[dict[str, float | str]],
) -> str:
    lines = [
        "# Learning-to-Rank Feature Diagnostics",
        "",
        "These values describe fitted model behavior; they are not causal explanations.",
        "",
        f"Selected model: `{selected_config.name}`.",
        "",
    ]
    if selected_config.family != "logistic_regression":
        lines.extend(
            [
                "HistGradientBoostingClassifier does not expose native impurity feature "
                "importance. No unsupported importance values are reported.",
                "",
                "The best validation logistic-regression coefficients are shown separately "
                "below and are not substituted as importance for the selected tree model.",
            ]
        )
    diagnostics = final_diagnostics or validation_diagnostics
    if diagnostics:
        lines.extend(
            [
                "",
                "| Feature | Standardized coefficient | Absolute coefficient |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in diagnostics:
            lines.append(
                f"| {row['feature']} | {float(row['coefficient']):.6f} | "
                f"{float(row['absolute_coefficient']):.6f} |"
            )
    else:
        lines.extend(["", "No native feature diagnostics are available for this model."])
    return "\n".join(lines) + "\n"


def _baseline_summaries(
    matrix: CandidateFeatureMatrix,
) -> dict[str, dict[str, Any]]:
    index = {name: position for position, name in enumerate(matrix.feature_names)}
    features = matrix.features
    scores = {
        "original_order": -features[:, index["candidate_position"]],
        "global_popularity": features[:, index["global_click_count"]],
        "time_decayed_popularity": features[:, index["time_decayed_click_count"]],
        "category_affinity": (
            features[:, index["category_history_count"]]
            + 0.5 * features[:, index["subcategory_history_count"]]
        ),
        "tfidf_content_similarity": features[
            :, index["tfidf_history_similarity"]
        ],
    }
    return {
        name: {
            "baseline_family": name,
            **evaluate_matrix_scores(matrix, np.asarray(values, dtype=np.float64)),
        }
        for name, values in scores.items()
    }


def _is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_key = (
        _metric(candidate, SELECTION_METRIC),
        _metric(candidate, "mrr"),
    )
    incumbent_key = (
        _metric(incumbent, SELECTION_METRIC),
        _metric(incumbent, "mrr"),
    )
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key
    return str(candidate["model_name"]) < str(incumbent["model_name"])


def _metric(summary: dict[str, Any], name: str) -> float:
    value = summary["metrics"].get(name)
    return float(value) if value is not None else float("-inf")


def _comparison_table(document: dict[str, Any]) -> str:
    rows: list[tuple[str, dict[str, Any]]] = [
        *(document["baselines"].items()),
        *(document["rankers"].items()),
    ]
    lines = [
        "| Method | MRR | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in rows:
        metrics = summary["metrics"]
        lines.append(
            f"| {name} | {_format(metrics.get('mrr'))} | "
            f"{_format(metrics.get('ndcg@5'))} | "
            f"{_format(metrics.get('ndcg@10'))} | "
            f"{_format(metrics.get('recall@10'))} | "
            f"{_format(metrics.get('hit_rate@10'))} | "
            f"{_format(metrics.get('auc'))} |"
        )
    return "\n".join(lines)


def _write_predictions(
    matrix: CandidateFeatureMatrix,
    scores: np.ndarray,
    path: Path,
    *,
    model_name: str,
    model_family: str,
    impression_chunk_size: int = 5000,
) -> None:
    schema = pa.schema(
        [
            pa.field("partition", pa.string()),
            pa.field("impression_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("candidate_news_id", pa.string()),
            pa.field("candidate_position", pa.int32()),
            pa.field("click_label", pa.int8()),
            pa.field("model_name", pa.string()),
            pa.field("model_family", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("predicted_rank", pa.int32()),
        ]
    )
    with pq.ParquetWriter(path, schema) as writer:
        for chunk_start in range(0, len(matrix.behaviors), impression_chunk_size):
            rows: list[dict[str, Any]] = []
            chunk = matrix.behaviors[chunk_start : chunk_start + impression_chunk_size]
            for local_index, behavior in enumerate(chunk, start=chunk_start):
                start = int(matrix.offsets[local_index])
                end = int(matrix.offsets[local_index + 1])
                candidates = [
                    ScoredCandidate(
                        impression_id=behavior.impression_id,
                        candidate_position=candidate.position,
                        click_label=candidate.clicked,
                        score=float(score),
                    )
                    for candidate, score in zip(
                        behavior.candidates,
                        scores[start:end],
                        strict=True,
                    )
                ]
                ranks = predicted_ranks(candidates)
                for candidate, score in zip(
                    behavior.candidates,
                    scores[start:end],
                    strict=True,
                ):
                    rows.append(
                        {
                            "partition": behavior.partition,
                            "impression_id": behavior.impression_id,
                            "user_id": behavior.user_id,
                            "timestamp": behavior.timestamp,
                            "candidate_news_id": candidate.news_id,
                            "candidate_position": candidate.position,
                            "click_label": candidate.clicked,
                            "model_name": model_name,
                            "model_family": model_family,
                            "score": float(score),
                            "predicted_rank": ranks[candidate.position],
                        }
                    )
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _without(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _peak_memory_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage / 1024.0)
