from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.evaluation.candidates import behaviors_to_candidate_rows
from feed_ranking_ops.evaluation.processed import (
    ProcessedDataError,
    load_news,
    load_processed_dataset,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


def _prepared(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "reports")
    return processed_dir


def test_processed_parquet_loading_and_candidate_explosion(tmp_path: Path):
    processed_dir = _prepared(tmp_path)

    dataset = load_processed_dataset(processed_dir)
    rows = behaviors_to_candidate_rows(dataset.behaviors["train"], dataset.news)

    assert len(dataset.news) == 5
    assert len(dataset.behaviors["train"]) == 4
    assert rows[0].impression_id == "1"
    assert rows[0].candidate_news_id == "N1"
    assert rows[0].click_label == 1
    assert rows[1].candidate_news_id == "N2"
    assert rows[1].history_news_ids == []
    assert [row.candidate_position for row in rows[:2]] == [0, 1]
    assert dataset.behaviors["train"][2].history_news_ids == ["N1", "N2"]


def test_processed_loader_can_limit_each_behavior_partition(tmp_path: Path):
    dataset = load_processed_dataset(
        _prepared(tmp_path),
        limit_impressions=1,
    )

    assert {name: len(rows) for name, rows in dataset.behaviors.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_missing_processed_file_raises(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    (processed_dir / "news.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="Missing processed Milestone 1 files"):
        load_processed_dataset(processed_dir)


def test_missing_news_column_raises(tmp_path: Path):
    path = tmp_path / "news.parquet"
    pq.write_table(pa.table({"news_id": ["N1"]}), path)

    with pytest.raises(ProcessedDataError, match="missing required columns"):
        load_news(path)


def test_candidate_missing_news_metadata_raises(tmp_path: Path):
    processed_dir = _prepared(tmp_path)
    news_table = pq.read_table(processed_dir / "news.parquet").to_pydict()
    keep = [news_id != "N2" for news_id in news_table["news_id"]]
    filtered = {
        key: [value for value, include in zip(values, keep, strict=True) if include]
        for key, values in news_table.items()
    }
    pq.write_table(pa.table(filtered), processed_dir / "news.parquet")

    with pytest.raises(ProcessedDataError, match="Candidates with missing news metadata"):
        load_processed_dataset(processed_dir)
