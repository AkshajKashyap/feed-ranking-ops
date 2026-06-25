import json
from pathlib import Path

import pyarrow.parquet as pq

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.evaluation.protocol import run_baseline_protocol
from feed_ranking_ops.evaluation.run_baselines import main as run_baselines_main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


def _prepared(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "prepare_reports")
    return processed_dir


def test_baseline_protocol_writes_reports_and_predictions(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "baseline_reports"

    result = run_baseline_protocol(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        limit_impressions=2,
        half_life_hours=[6, 24],
    )

    expected_outputs = [
        "validation_metrics.json",
        "test_metrics.json",
        "model_comparison.md",
        "protocol.json",
        "validation_predictions.parquet",
        "test_predictions.parquet",
    ]
    for filename in expected_outputs:
        assert (reports_dir / filename).is_file()

    validation_predictions = pq.read_table(reports_dir / "validation_predictions.parquet")
    test_predictions = pq.read_table(reports_dir / "test_predictions.parquet")
    assert validation_predictions.num_rows > 0
    assert test_predictions.num_rows > 0
    assert {
        "impression_id",
        "candidate_news_id",
        "candidate_position",
        "click_label",
        "baseline_name",
        "score",
        "predicted_rank",
        "partition",
    }.issubset(set(test_predictions.column_names))
    assert result["protocol"]["smoke_test"] is True
    assert result["protocol"]["test_labels_used_for_selection"] is False


def test_protocol_is_deterministic(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    first_dir = tmp_path / "reports_a"
    second_dir = tmp_path / "reports_b"

    first = run_baseline_protocol(
        processed_dir=processed_dir,
        reports_dir=first_dir,
        limit_impressions=2,
        half_life_hours=[6, 24],
    )
    second = run_baseline_protocol(
        processed_dir=processed_dir,
        reports_dir=second_dir,
        limit_impressions=2,
        half_life_hours=[6, 24],
    )

    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["test_metrics"] == second["test_metrics"]
    assert (first_dir / "validation_metrics.json").read_text(encoding="utf-8") == (
        second_dir / "validation_metrics.json"
    ).read_text(encoding="utf-8")


def test_protocol_records_validation_and_test_isolation(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "baseline_reports"

    run_baseline_protocol(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        limit_impressions=2,
        half_life_hours=[6, 24],
    )

    protocol = json.loads((reports_dir / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (reports_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    test = json.loads((reports_dir / "test_metrics.json").read_text(encoding="utf-8"))

    assert protocol["validation_fit_partitions"] == ["train"]
    assert protocol["test_fit_partitions"] == ["train", "validation"]
    assert protocol["selected_hyperparameters"]
    for summary in validation["baselines"].values():
        assert summary["fit_metadata"]["fitting_partitions"] == ["train"]
    for summary in test["baselines"].values():
        assert summary["fit_metadata"]["fitting_partitions"] == ["train", "validation"]


def test_run_baselines_cli_smoke(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "cli_reports"

    exit_code = run_baselines_main(
        [
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--limit-impressions",
            "2",
            "--half-lives",
            "6,24",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "model_comparison.md").is_file()
    assert (reports_dir / "test_predictions.parquet").is_file()
