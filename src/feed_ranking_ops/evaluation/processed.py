from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from feed_ranking_ops.data.schemas import MindDataError

PROCESSED_FILES = {
    "news": "news.parquet",
    "train": "train_behaviors.parquet",
    "validation": "validation_behaviors.parquet",
    "test": "test_behaviors.parquet",
    "metadata": "split_metadata.json",
}

NEWS_COLUMNS = {
    "source_split",
    "source_row_number",
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
}

BEHAVIOR_COLUMNS = {
    "source_split",
    "source_row_number",
    "source_row_hash",
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
}


class ProcessedDataError(MindDataError):
    """Raised when processed Milestone 1 artifacts are missing or malformed."""


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass(frozen=True)
class ImpressionCandidate:
    position: int
    news_id: str
    clicked: int | None


@dataclass(frozen=True)
class BehaviorImpression:
    partition: str
    impression_id: str
    user_id: str
    timestamp: datetime
    history_news_ids: list[str]
    candidates: list[ImpressionCandidate]


@dataclass(frozen=True)
class ProcessedDataset:
    news: dict[str, NewsItem]
    behaviors: dict[str, list[BehaviorImpression]]
    split_metadata: dict[str, Any]
    history_missing_counts: dict[str, int]


def load_processed_dataset(
    processed_dir: Path,
    *,
    limit_impressions: int | None = None,
) -> ProcessedDataset:
    """Load and validate Milestone 1 processed artifacts."""
    if limit_impressions is not None and limit_impressions <= 0:
        raise ValueError("limit_impressions must be positive when provided")
    _require_processed_files(processed_dir)
    news = load_news(processed_dir / PROCESSED_FILES["news"])
    behaviors = {
        partition: load_behavior_partition(
            processed_dir / PROCESSED_FILES[partition],
            partition=partition,
            limit_rows=limit_impressions,
        )
        for partition in ("train", "validation", "test")
    }
    split_metadata = json.loads(
        (processed_dir / PROCESSED_FILES["metadata"]).read_text(encoding="utf-8")
    )
    history_missing_counts = _validate_news_references(news, behaviors)
    return ProcessedDataset(
        news=news,
        behaviors=behaviors,
        split_metadata=split_metadata,
        history_missing_counts=history_missing_counts,
    )


def load_news(path: Path) -> dict[str, NewsItem]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed news file: {path}")
    table = pq.read_table(path)
    _require_columns(path, table.column_names, NEWS_COLUMNS)
    rows = table.to_pylist()
    news: dict[str, NewsItem] = {}
    for row_index, row in enumerate(rows, start=1):
        news_id = row.get("news_id")
        if not isinstance(news_id, str) or not news_id:
            raise ProcessedDataError(f"{path}: row {row_index} has invalid news_id")
        if news_id in news:
            raise ProcessedDataError(f"{path}: duplicate news_id {news_id!r}")
        news[news_id] = NewsItem(
            news_id=news_id,
            category=_string_or_empty(row.get("category")),
            subcategory=_string_or_empty(row.get("subcategory")),
            title=_string_or_empty(row.get("title")),
            abstract=_string_or_empty(row.get("abstract")),
        )
    return news


def load_behavior_partition(
    path: Path,
    *,
    partition: str,
    limit_rows: int | None = None,
) -> list[BehaviorImpression]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed behavior file: {path}")
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be positive when provided")
    parquet_file = pq.ParquetFile(path)
    _require_columns(path, parquet_file.schema_arrow.names, BEHAVIOR_COLUMNS)
    if limit_rows is None:
        rows = parquet_file.read().to_pylist()
    else:
        first_batch = next(
            parquet_file.iter_batches(batch_size=limit_rows),
            None,
        )
        rows = [] if first_batch is None else first_batch.to_pylist()
    seen_impression_ids: set[str] = set()
    behaviors: list[BehaviorImpression] = []
    for row_index, row in enumerate(rows[:limit_rows], start=1):
        impression_id = row.get("impression_id")
        user_id = row.get("user_id")
        timestamp = row.get("timestamp")
        if not isinstance(impression_id, str) or not impression_id:
            raise ProcessedDataError(f"{path}: row {row_index} has invalid impression_id")
        if impression_id in seen_impression_ids:
            raise ProcessedDataError(
                f"{path}: duplicate impression_id {impression_id!r} in {partition}"
            )
        seen_impression_ids.add(impression_id)
        if not isinstance(user_id, str) or not user_id:
            raise ProcessedDataError(f"{path}: row {row_index} has invalid user_id")
        if not isinstance(timestamp, datetime):
            raise ProcessedDataError(f"{path}: row {row_index} has malformed timestamp")
        history = _validate_history(path, row_index, row.get("history"))
        candidates = _validate_candidates(path, row_index, row.get("impressions"))
        behaviors.append(
            BehaviorImpression(
                partition=partition,
                impression_id=impression_id,
                user_id=user_id,
                timestamp=timestamp,
                history_news_ids=history,
                candidates=candidates,
            )
        )
    return behaviors


def _require_processed_files(processed_dir: Path) -> None:
    missing = [
        processed_dir / filename
        for filename in PROCESSED_FILES.values()
        if not (processed_dir / filename).is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing processed Milestone 1 files:\n{formatted}")


def _require_columns(path: Path, actual: list[str], required: set[str]) -> None:
    missing = sorted(required.difference(actual))
    if missing:
        raise ProcessedDataError(
            f"{path}: missing required columns: {', '.join(missing)}"
        )


def _validate_history(path: Path, row_index: int, value: Any) -> list[str]:
    if value is None:
        raise ProcessedDataError(f"{path}: row {row_index} has malformed history")
    if not isinstance(value, list):
        raise ProcessedDataError(f"{path}: row {row_index} history must be a list")
    if not all(isinstance(news_id, str) and news_id for news_id in value):
        raise ProcessedDataError(
            f"{path}: row {row_index} history must contain non-empty strings"
        )
    return list(value)


def _validate_candidates(
    path: Path,
    row_index: int,
    value: Any,
) -> list[ImpressionCandidate]:
    if value is None or not isinstance(value, list):
        raise ProcessedDataError(
            f"{path}: row {row_index} impressions must be a list of candidate structs"
        )
    candidates: list[ImpressionCandidate] = []
    seen_positions: set[int] = set()
    for candidate_index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            raise ProcessedDataError(
                f"{path}: row {row_index} candidate {candidate_index} is malformed"
            )
        position = candidate.get("position")
        news_id = candidate.get("news_id")
        clicked = candidate.get("clicked")
        if not isinstance(position, int) or position < 0:
            raise ProcessedDataError(
                f"{path}: row {row_index} candidate {candidate_index} has invalid position"
            )
        if position in seen_positions:
            raise ProcessedDataError(
                f"{path}: row {row_index} has duplicate candidate position {position}"
            )
        seen_positions.add(position)
        if not isinstance(news_id, str) or not news_id:
            raise ProcessedDataError(
                f"{path}: row {row_index} candidate {candidate_index} has invalid news_id"
            )
        if clicked not in {None, 0, 1}:
            raise ProcessedDataError(
                f"{path}: row {row_index} candidate {candidate_index} has invalid label"
            )
        candidates.append(
            ImpressionCandidate(position=position, news_id=news_id, clicked=clicked)
        )
    positions = [candidate.position for candidate in candidates]
    if positions != sorted(positions):
        raise ProcessedDataError(
            f"{path}: row {row_index} candidate positions are not in source order"
        )
    return candidates


def _validate_news_references(
    news: dict[str, NewsItem],
    behaviors: dict[str, list[BehaviorImpression]],
) -> dict[str, int]:
    history_missing_counts: dict[str, int] = {}
    missing_candidates: dict[str, list[str]] = {}
    for partition, rows in behaviors.items():
        history_missing_counts[partition] = sum(
            1
            for row in rows
            for news_id in row.history_news_ids
            if news_id not in news
        )
        missing_candidate_ids = sorted(
            {
                candidate.news_id
                for row in rows
                for candidate in row.candidates
                if candidate.news_id not in news
            }
        )
        if missing_candidate_ids:
            missing_candidates[partition] = missing_candidate_ids
    if missing_candidates:
        details = "; ".join(
            f"{partition}: {', '.join(news_ids[:20])}"
            for partition, news_ids in sorted(missing_candidates.items())
        )
        raise ProcessedDataError(f"Candidates with missing news metadata: {details}")
    return history_missing_counts


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""
