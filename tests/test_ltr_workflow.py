import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.ranking.models import RankerConfig
from feed_ranking_ops.ranking.protocol import run_ltr_protocol
from feed_ranking_ops.ranking.run_ltr import main as run_ltr_main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"
FAST_CONFIGS = [
    RankerConfig("logistic_regression", {"C": 0.1}),
    RankerConfig(
        "hist_gradient_boosting",
        {"learning_rate": 0.1, "max_leaf_nodes": 7},
    ),
]


def _prepared(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "prepare_reports")
    return processed_dir


def test_ltr_protocol_writes_reports_and_compatible_predictions(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "ltr"

    result = run_ltr_protocol(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        limit_impressions=2,
        ranker_configs=FAST_CONFIGS,
    )

    expected = {
        "validation_metrics.json",
        "test_metrics.json",
        "protocol.json",
        "model_comparison.md",
        "feature_importance.md",
        "validation_predictions.parquet",
        "test_predictions.parquet",
    }
    assert expected.issubset({path.name for path in reports_dir.iterdir()})
    predictions = pq.read_table(reports_dir / "test_predictions.parquet")
    assert predictions.num_rows > 0
    assert {
        "partition",
        "impression_id",
        "candidate_news_id",
        "candidate_position",
        "click_label",
        "model_name",
        "score",
        "predicted_rank",
    }.issubset(predictions.column_names)
    assert np.isfinite(predictions.column("score").to_numpy()).all()
    assert result["protocol"]["limit_impressions"] == 2
    assert result["protocol"]["partition_sizes"] == {
        "train": 2,
        "validation": 1,
        "test": 2,
    }


def test_ltr_protocol_records_strict_partition_boundaries(tmp_path: Path):
    reports_dir = tmp_path / "ltr"
    run_ltr_protocol(
        processed_dir=_prepared(tmp_path),
        reports_dir=reports_dir,
        limit_impressions=2,
        ranker_configs=FAST_CONFIGS,
    )

    protocol = json.loads((reports_dir / "protocol.json").read_text(encoding="utf-8"))

    assert protocol["validation_fit_partitions"] == ["train"]
    assert protocol["test_fit_partitions"] == ["train", "validation"]
    assert protocol["validation_labels_used_for_model_fitting"] is False
    assert protocol["test_labels_used_for_selection"] is False
    assert protocol["feature_metadata"]["validation"]["fitting_partitions"] == ["train"]
    assert protocol["feature_metadata"]["test"]["fitting_partitions"] == [
        "train",
        "validation",
    ]
    assert protocol["internal_holdout"] == "official_dev_test"
    assert protocol["timing"]["total_runtime_seconds"] > 0


def test_ltr_predictions_and_metrics_are_deterministic(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_ltr_protocol(
        processed_dir=processed_dir,
        reports_dir=first_dir,
        limit_impressions=2,
        seed=7,
        ranker_configs=FAST_CONFIGS,
    )
    second = run_ltr_protocol(
        processed_dir=processed_dir,
        reports_dir=second_dir,
        limit_impressions=2,
        seed=7,
        ranker_configs=FAST_CONFIGS,
    )

    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["test_metrics"] == second["test_metrics"]
    first_predictions = pq.read_table(first_dir / "test_predictions.parquet")
    second_predictions = pq.read_table(second_dir / "test_predictions.parquet")
    assert first_predictions.equals(second_predictions)


def test_model_comparison_contains_baselines_and_learned_ranker(tmp_path: Path):
    reports_dir = tmp_path / "ltr"
    run_ltr_protocol(
        processed_dir=_prepared(tmp_path),
        reports_dir=reports_dir,
        limit_impressions=2,
        ranker_configs=FAST_CONFIGS,
    )

    report = (reports_dir / "model_comparison.md").read_text(encoding="utf-8")

    assert "original_order" in report
    assert "category_affinity" in report
    assert "tfidf_content_similarity" in report
    assert "logistic_regression" in report
    assert "Internal Test Results" in report


def test_run_ltr_cli_smoke(tmp_path: Path):
    reports_dir = tmp_path / "ltr"

    exit_code = run_ltr_main(
        [
            "--processed-dir",
            str(_prepared(tmp_path)),
            "--reports-dir",
            str(reports_dir),
            "--limit-impressions",
            "2",
            "--seed",
            "9",
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "model_comparison.md").is_file()
    assert (reports_dir / "test_predictions.parquet").is_file()
