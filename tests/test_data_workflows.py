import json
from pathlib import Path

import pyarrow.parquet as pq

from feed_ranking_ops.data.audit_dataset import audit_dataset
from feed_ranking_ops.data.layout import validate_mind_layout
from feed_ranking_ops.data.prepare_dataset import prepare_dataset


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


def test_layout_validation_reports_missing_source_files(tmp_path: Path):
    result = validate_mind_layout(tmp_path)

    assert not result.is_valid
    assert set(result.missing_files) == {
        "train_news",
        "train_behaviors",
        "dev_news",
        "dev_behaviors",
    }


def test_audit_report_generation(tmp_path: Path):
    reports_dir = tmp_path / "reports"

    audit = audit_dataset(FIXTURE_DIR, reports_dir)

    assert (reports_dir / "data_audit.json").exists()
    assert (reports_dir / "data_audit.md").exists()
    assert audit["news"]["duplicate_news_id_count"] == 1
    assert audit["behaviors"]["empty_history_count"] == 1
    assert audit["behaviors"]["clicked_candidate_count"] == 5
    assert audit["behaviors"]["unlabeled_candidate_count"] == 2


def test_chronological_split_outputs_parquet_and_metadata(tmp_path: Path):
    output_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    metadata = prepare_dataset(FIXTURE_DIR, output_dir, reports_dir)

    assert metadata["partitions"]["train"]["row_count"] == 4
    assert metadata["partitions"]["validation"]["row_count"] == 1
    assert metadata["partitions"]["test"]["row_count"] == 2
    assert metadata["partitions"]["train"]["timestamp_max"] <= metadata["partitions"][
        "validation"
    ]["timestamp_min"]
    assert metadata["integrity_checks"]["no_impression_id_overlap"] is True
    assert metadata["integrity_checks"]["official_dev_rows_never_in_train_or_validation"] is True
    assert metadata["integrity_checks"]["history_ordering_preserved"] is True

    expected_outputs = [
        "news.parquet",
        "train_behaviors.parquet",
        "validation_behaviors.parquet",
        "test_behaviors.parquet",
        "split_metadata.json",
    ]
    for filename in expected_outputs:
        assert (output_dir / filename).exists()
    assert (reports_dir / "split_summary.md").exists()

    train_table = pq.read_table(output_dir / "train_behaviors.parquet")
    assert train_table.num_rows == 4
    assert str(train_table.schema.field("history").type) == "list<element: string>"
    assert "impressions" in train_table.column_names


def test_split_determinism(tmp_path: Path):
    first_output = tmp_path / "processed_a"
    second_output = tmp_path / "processed_b"

    first_metadata = prepare_dataset(FIXTURE_DIR, first_output, tmp_path / "reports_a")
    second_metadata = prepare_dataset(FIXTURE_DIR, second_output, tmp_path / "reports_b")

    assert first_metadata == second_metadata
    assert json.loads((first_output / "split_metadata.json").read_text(encoding="utf-8")) == (
        json.loads((second_output / "split_metadata.json").read_text(encoding="utf-8"))
    )


def test_no_partition_overlap(tmp_path: Path):
    metadata = prepare_dataset(FIXTURE_DIR, tmp_path / "processed", tmp_path / "reports")
    overlaps = metadata["integrity_checks"]["impression_id_overlaps"]

    assert overlaps == {
        "test__train": [],
        "test__validation": [],
        "train__validation": [],
    }
