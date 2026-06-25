import json
from pathlib import Path

import pyarrow.parquet as pq

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.retrieval.protocol import run_exact_retrieval_protocol
from feed_ranking_ops.retrieval.run_exact_retrieval import main as retrieval_main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


def _prepared(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "prepare_reports")
    return processed_dir


def test_exact_retrieval_protocol_writes_reports_and_predictions(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "retrieval"

    result = run_exact_retrieval_protocol(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        limit_queries=2,
        text_configs=["title", "title_abstract"],
        history_lengths=[10, None],
        decay_values=[0.5],
        top_k=20,
    )

    expected = [
        "validation_metrics.json",
        "test_metrics.json",
        "config_sweep.csv",
        "model_comparison.md",
        "protocol.json",
        "availability_summary.json",
        "validation_retrievals.parquet",
        "test_retrievals.parquet",
    ]
    for filename in expected:
        assert (reports_dir / filename).is_file()

    table = pq.read_table(reports_dir / "test_retrievals.parquet")
    assert {
        "partition",
        "impression_id",
        "user_id",
        "query_timestamp",
        "retrieved_rank",
        "retrieved_news_id",
        "score",
        "is_clicked_target",
        "was_in_history",
        "catalog_size",
        "fallback_used",
        "selected_configuration",
    }.issubset(set(table.column_names))
    assert result["protocol"]["test_labels_used_for_selection"] is False
    assert result["protocol"]["catalog_protocol"] == "observed_available"


def test_retrieval_protocol_is_deterministic_and_isolated(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_exact_retrieval_protocol(
        processed_dir=processed_dir,
        reports_dir=first,
        limit_queries=2,
        text_configs=["title"],
        history_lengths=[10],
        decay_values=[0.5],
        top_k=20,
    )
    second_result = run_exact_retrieval_protocol(
        processed_dir=processed_dir,
        reports_dir=second,
        limit_queries=2,
        text_configs=["title"],
        history_lengths=[10],
        decay_values=[0.5],
        top_k=20,
    )

    assert first_result["protocol"] == second_result["protocol"]
    assert first_result["validation_metrics"]["metrics"] == second_result[
        "validation_metrics"
    ]["metrics"]
    protocol = json.loads((first / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((first / "validation_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((first / "test_metrics.json").read_text(encoding="utf-8"))
    assert protocol["validation_fit_partitions"] == ["train"]
    assert protocol["test_fit_partitions"] == ["train", "validation"]
    assert validation["fit_metadata"]["fitting_partitions"] == ["train"]
    assert test["fit_metadata"]["fitting_partitions"] == ["train", "validation"]


def test_exact_retrieval_cli_smoke(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "cli_retrieval"

    exit_code = retrieval_main(
        [
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--limit-queries",
            "2",
            "--top-k",
            "20",
            "--text-configs",
            "title",
            "--history-lengths",
            "10",
            "--decay-values",
            "0.5",
        ]
    )

    assert exit_code == 0
    assert (reports_dir / "model_comparison.md").is_file()
    assert (reports_dir / "test_retrievals.parquet").is_file()
