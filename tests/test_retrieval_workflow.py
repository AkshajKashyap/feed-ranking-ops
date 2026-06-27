import json
from pathlib import Path

import pyarrow.parquet as pq

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.retrieval import protocol as retrieval_protocol
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
    assert result["validation_metrics"]["metrics"]["recall@100"] == 1.0
    assert result["test_metrics"]["metrics"]["mrr@100"] == 0.25
    assert result["validation_metrics"]["n_queries"] == 1
    assert result["test_metrics"]["n_queries"] == 2

    timing = result["protocol"]["timing"]
    assert timing["number_of_queries"] == 3
    assert timing["number_of_articles"] == 5
    assert timing["article_vectorization_seconds"] >= 0
    assert timing["query_construction_seconds"] >= 0
    assert timing["scoring_seconds"] >= 0
    assert timing["metric_evaluation_seconds"] >= 0
    assert timing["total_runtime_seconds"] >= 0
    assert timing["average_eligible_articles_per_test_query"] is not None

    report = (reports_dir / "model_comparison.md").read_text(encoding="utf-8")
    assert "End-to-end timing" in report


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

    first_protocol = {
        key: value for key, value in first_result["protocol"].items() if key != "timing"
    }
    second_protocol = {
        key: value for key, value in second_result["protocol"].items() if key != "timing"
    }
    assert first_protocol == second_protocol
    assert first_result["validation_metrics"]["metrics"] == second_result[
        "validation_metrics"
    ]["metrics"]
    for filename in ["validation_retrievals.parquet", "test_retrievals.parquet"]:
        columns = [
            "impression_id",
            "retrieved_rank",
            "retrieved_news_id",
            "score",
        ]
        first_retrievals = pq.read_table(first / filename, columns=columns)
        second_retrievals = pq.read_table(second / filename, columns=columns)
        assert first_retrievals.equals(second_retrievals)
    protocol = json.loads((first / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((first / "validation_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((first / "test_metrics.json").read_text(encoding="utf-8"))
    assert protocol["validation_fit_partitions"] == ["train"]
    assert protocol["test_fit_partitions"] == ["train", "validation"]
    assert validation["fit_metadata"]["fitting_partitions"] == ["train"]
    assert test["fit_metadata"]["fitting_partitions"] == ["train", "validation"]


def test_exact_retrieval_preserves_expected_synthetic_ordering(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    reports_dir = tmp_path / "retrieval"

    run_exact_retrieval_protocol(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        limit_queries=2,
        text_configs=["title"],
        history_lengths=[10],
        decay_values=[0.5],
        top_k=20,
    )

    validation = pq.read_table(
        reports_dir / "validation_retrievals.parquet",
        columns=["impression_id", "retrieved_rank", "retrieved_news_id"],
    ).to_pylist()
    test = pq.read_table(
        reports_dir / "test_retrievals.parquet",
        columns=["impression_id", "retrieved_rank", "retrieved_news_id"],
    ).to_pylist()

    assert validation == [
        {"impression_id": "5", "retrieved_rank": 1, "retrieved_news_id": "N2"},
        {"impression_id": "5", "retrieved_rank": 2, "retrieved_news_id": "N4"},
    ]
    assert test == [
        {"impression_id": "6", "retrieved_rank": 1, "retrieved_news_id": "N3"},
        {"impression_id": "6", "retrieved_rank": 2, "retrieved_news_id": "N4"},
        {"impression_id": "6", "retrieved_rank": 3, "retrieved_news_id": "N5"},
        {"impression_id": "7", "retrieved_rank": 1, "retrieved_news_id": "N1"},
        {"impression_id": "7", "retrieved_rank": 2, "retrieved_news_id": "N2"},
        {"impression_id": "7", "retrieved_rank": 3, "retrieved_news_id": "N4"},
        {"impression_id": "7", "retrieved_rank": 4, "retrieved_news_id": "N5"},
    ]


def test_article_vectorization_is_reused_across_profile_configurations(
    tmp_path: Path,
    monkeypatch,
):
    processed_dir = _prepared(tmp_path)
    original_fit = retrieval_protocol.fit_article_text_index
    calls = []

    def counting_fit(**kwargs):
        calls.append(kwargs["text_config"].name)
        return original_fit(**kwargs)

    monkeypatch.setattr(retrieval_protocol, "fit_article_text_index", counting_fit)

    run_exact_retrieval_protocol(
        processed_dir=processed_dir,
        reports_dir=tmp_path / "retrieval",
        limit_queries=2,
        text_configs=["title"],
        history_lengths=[10, None],
        decay_values=[0.5],
        top_k=20,
    )

    assert calls == ["title", "title"]


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
