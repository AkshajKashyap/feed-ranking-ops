import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from feed_ranking_ops.data.audit_dataset import audit_dataset
from feed_ranking_ops.data.layout import require_valid_mind_layout, validate_mind_layout
from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.data.schemas import MindDataError


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"
PROCESSED_FILENAMES = {
    "news.parquet",
    "train_behaviors.parquet",
    "validation_behaviors.parquet",
    "test_behaviors.parquet",
    "split_metadata.json",
}


def _train_only_fixture(tmp_path: Path, *, behavior_rows: int = 20) -> Path:
    data_dir = tmp_path / "raw"
    train_dir = data_dir / "MINDsmall_train"
    shutil.copytree(FIXTURE_DIR / "MINDsmall_train", train_dir)
    rows = [
        f"{index}\tU{index}\t11/15/2019 8:00:00 AM\tN1 N2\tN1-1 N2-0"
        for index in range(1, behavior_rows + 1)
    ]
    (train_dir / "behaviors.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return data_dir


def test_layout_validation_reports_missing_source_files(tmp_path: Path):
    result = validate_mind_layout(tmp_path)

    assert not result.is_valid
    assert set(result.missing_files) == {
        "train_news",
        "train_behaviors",
        "dev_news",
        "dev_behaviors",
    }


def test_official_protocol_requires_dev_files(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)

    result = validate_mind_layout(data_dir, protocol="official_train_dev")

    assert not result.is_valid
    assert set(result.missing_files) == {"dev_news", "dev_behaviors"}
    with pytest.raises(FileNotFoundError, match="official_train_dev"):
        require_valid_mind_layout(data_dir, protocol="official_train_dev")


def test_train_only_layout_succeeds_without_dev_files(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)

    result = validate_mind_layout(data_dir, protocol="train_only_chronological")

    assert result.is_valid
    assert set(result.existing_files) == {"train_news", "train_behaviors"}
    assert result.protocol == "train_only_chronological"


def test_audit_report_generation(tmp_path: Path):
    reports_dir = tmp_path / "reports"

    audit = audit_dataset(FIXTURE_DIR, reports_dir)

    assert (reports_dir / "data_audit.json").exists()
    assert (reports_dir / "data_audit.md").exists()
    assert audit["news"]["duplicate_news_id_count"] == 1
    assert audit["behaviors"]["empty_history_count"] == 1
    assert audit["behaviors"]["clicked_candidate_count"] == 5
    assert audit["behaviors"]["unlabeled_candidate_count"] == 2


def test_train_only_audit_uses_no_dev_statistics(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)
    reports_dir = tmp_path / "reports"

    audit = audit_dataset(
        data_dir,
        reports_dir,
        protocol="train_only_chronological",
    )

    assert audit["protocol"] == "train_only_chronological"
    assert audit["source_splits_used"] == ["train"]
    assert audit["official_dev_used"] is False
    assert audit["news"]["row_count_by_source_split"] == {"train": 4}
    assert audit["behaviors"]["row_count_by_source_split"] == {"train": 20}
    markdown = (reports_dir / "data_audit.md").read_text(encoding="utf-8")
    assert "No official dev split was used or fabricated" in markdown


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


def test_train_only_split_writes_compatible_70_15_15_outputs(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)
    output_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    metadata = prepare_dataset(
        data_dir,
        output_dir,
        reports_dir,
        protocol="train_only_chronological",
    )

    assert metadata["protocol"] == "train_only_chronological"
    assert metadata["source_splits_used"] == ["train"]
    assert metadata["requested_ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert metadata["observed_row_counts"] == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    assert metadata["observed_ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert metadata["final_partition_type"] == "internal_chronological_holdout"
    assert "not directly comparable" in metadata["comparability_warning"]
    assert metadata["integrity_checks"]["validation_max_not_later_than_test_min"] is True
    assert metadata["integrity_checks"]["no_impression_id_overlap"] is True
    assert metadata["integrity_checks"]["no_source_row_overlap"] is True
    assert metadata["integrity_checks"]["history_ordering_preserved"] is True
    assert metadata["integrity_checks"]["candidate_labels_preserved"] is True
    assert metadata["integrity_checks"]["random_splitting_used"] is False
    assert {path.name for path in output_dir.iterdir()} == PROCESSED_FILENAMES
    assert (reports_dir / "split_summary.md").is_file()


def test_train_only_tied_timestamps_use_stable_source_row_boundaries(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = prepare_dataset(
        data_dir,
        first_output,
        tmp_path / "reports_first",
        protocol="train_only_chronological",
    )
    second = prepare_dataset(
        data_dir,
        second_output,
        tmp_path / "reports_second",
        protocol="train_only_chronological",
    )

    expected_source_rows = {
        "train": list(range(1, 15)),
        "validation": list(range(15, 18)),
        "test": list(range(18, 21)),
    }
    for partition, expected in expected_source_rows.items():
        first_rows = pq.read_table(
            first_output / f"{partition}_behaviors.parquet",
            columns=["source_row_number"],
        ).column("source_row_number").to_pylist()
        second_rows = pq.read_table(
            second_output / f"{partition}_behaviors.parquet",
            columns=["source_row_number"],
        ).column("source_row_number").to_pylist()
        assert first_rows == expected
        assert second_rows == expected

    boundaries = first["chronological_boundary_timestamps"]
    assert boundaries["train_max_timestamp"] == boundaries["validation_min_timestamp"]
    assert boundaries["validation_max_timestamp"] == boundaries["test_min_timestamp"]
    assert first == second


def test_train_only_ratios_must_sum_to_one(tmp_path: Path):
    data_dir = _train_only_fixture(tmp_path)

    with pytest.raises(MindDataError, match="sum to 1.0"):
        prepare_dataset(
            data_dir,
            tmp_path / "processed",
            tmp_path / "reports",
            protocol="train_only_chronological",
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.2,
        )


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
