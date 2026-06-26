from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.data.layout import (
    DATA_PROTOCOLS,
    DEFAULT_DATA_PROTOCOL,
    EXPECTED_MIND_FILES,
    DataProtocol,
    require_valid_mind_layout,
)
from feed_ranking_ops.data.parsers import parse_behavior_file, parse_news_file
from feed_ranking_ops.data.schemas import BehaviorRecord, MindDataError, NewsRecord

DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_TRAIN_ONLY_TRAIN_RATIO = 0.70
DEFAULT_TRAIN_ONLY_VALIDATION_RATIO = 0.15
DEFAULT_TRAIN_ONLY_TEST_RATIO = 0.15
INTERNAL_HOLDOUT_WARNING = (
    "The final partition is an internal chronological holdout from MINDsmall_train. "
    "Its metrics are not directly comparable to official MIND validation results."
)

NEWS_SCHEMA_DESCRIPTION = {
    "source_split": "Original MIND source split: train or dev.",
    "source_row_number": "1-based row number within the original news.tsv file.",
    "news_id": "MIND news identifier.",
    "category": "MIND news category.",
    "subcategory": "MIND news subcategory.",
    "title": "News title; may be empty if source data is empty.",
    "abstract": "News abstract; may be empty if source data is empty.",
    "url": "Source URL from MIND.",
    "title_entities": "Raw MIND title entity JSON string.",
    "abstract_entities": "Raw MIND abstract entity JSON string.",
}

BEHAVIOR_SCHEMA_DESCRIPTION = {
    "source_split": "Original MIND source split.",
    "source_row_number": "1-based row number within the original behaviors.tsv file.",
    "source_row_hash": "SHA-256 hash of the original behavior TSV line.",
    "impression_id": "MIND impression identifier.",
    "user_id": "MIND user identifier.",
    "timestamp": "UTC-normalized timestamp parsed from the source row.",
    "history": "Ordered list<string> of previously clicked news IDs from the source history field.",
    "impressions": (
        "list<struct<position:int32, news_id:string, clicked:int8?>> preserving candidate "
        "order. clicked is 1, 0, or null for inference-style unlabeled candidates."
    ),
}


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    *,
    protocol: DataProtocol = DEFAULT_DATA_PROTOCOL,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    train_ratio: float = DEFAULT_TRAIN_ONLY_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_TRAIN_ONLY_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TRAIN_ONLY_TEST_RATIO,
) -> dict[str, Any]:
    if protocol == "official_train_dev" and not 0 < train_fraction < 1:
        raise MindDataError("--train-fraction must be greater than 0 and less than 1.")
    if protocol == "train_only_chronological":
        _validate_train_only_ratios(train_ratio, validation_ratio, test_ratio)

    require_valid_mind_layout(data_dir, protocol)

    train_news, _ = parse_news_file(data_dir / EXPECTED_MIND_FILES["train_news"], "train")
    train_behaviors, _ = parse_behavior_file(
        data_dir / EXPECTED_MIND_FILES["train_behaviors"], "train", strict=True
    )

    train_sorted = sorted(train_behaviors, key=_behavior_sort_key)
    if protocol == "official_train_dev":
        dev_news, _ = parse_news_file(data_dir / EXPECTED_MIND_FILES["dev_news"], "dev")
        dev_behaviors, _ = parse_behavior_file(
            data_dir / EXPECTED_MIND_FILES["dev_behaviors"], "dev", strict=True
        )
        train_count = _chronological_train_count(len(train_sorted), train_fraction)
        train_partition = train_sorted[:train_count]
        validation_partition = train_sorted[train_count:]
        test_partition = sorted(dev_behaviors, key=_behavior_sort_key)
        news_records = _dedupe_news([*train_news, *dev_news])
    else:
        train_end, validation_end = _train_only_boundaries(
            len(train_sorted),
            train_ratio,
            validation_ratio,
        )
        train_partition = train_sorted[:train_end]
        validation_partition = train_sorted[train_end:validation_end]
        test_partition = train_sorted[validation_end:]
        news_records = _dedupe_news(train_news)

    news_ids = {record.news_id for record in news_records}
    partitions = {
        "train": train_partition,
        "validation": validation_partition,
        "test": test_partition,
    }
    integrity = _integrity_checks(partitions, news_ids, protocol=protocol)
    _raise_for_failed_integrity(integrity, protocol=protocol)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_news_parquet(news_records, output_dir / "news.parquet")
    _write_behaviors_parquet(train_partition, output_dir / "train_behaviors.parquet")
    _write_behaviors_parquet(validation_partition, output_dir / "validation_behaviors.parquet")
    _write_behaviors_parquet(test_partition, output_dir / "test_behaviors.parquet")

    metadata = _build_metadata(
        protocol=protocol,
        train_fraction=train_fraction,
        requested_ratios={
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        partitions=partitions,
        integrity=integrity,
        news_records=news_records,
    )
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "split_summary.md").write_text(
        render_split_summary(metadata),
        encoding="utf-8",
    )

    return metadata


def render_split_summary(metadata: dict[str, Any]) -> str:
    partitions = metadata["partitions"]
    observed = metadata["observed_boundaries"]
    protocol = metadata["protocol"]
    final_partition_type = metadata["final_partition_type"]
    if protocol == "official_train_dev":
        description = (
            "Official MIND train behaviors are sorted chronologically and split into model "
            "train and validation partitions. Official MIND dev behaviors are kept as the "
            "offline test partition."
        )
    else:
        description = (
            "Only official MIND train behaviors are used. Rows are sorted chronologically "
            "and split into train, validation, and an internal chronological holdout."
        )
    lines = [
        "# MIND-small Temporal Split Summary",
        "",
        f"- Protocol: `{protocol}`",
        f"- Source splits used: {', '.join(metadata['source_splits_used'])}",
        f"- Final partition type: `{final_partition_type}`",
        "",
        description,
        "",
        "## Requested and Observed Ratios",
        "",
        "| Partition | Requested ratio | Observed rows | Observed ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("train", "validation", "test"):
        lines.append(
            f"| {name} | {_display_ratio(metadata['requested_ratios'][name])} | "
            f"{metadata['observed_row_counts'][name]} | "
            f"{metadata['observed_ratios'][name]:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Partition Counts",
            "",
            "| Partition | Rows | Timestamp min | Timestamp max | Source split |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for name in ("train", "validation", "test"):
        info = partitions[name]
        lines.append(
            f"| {name} | {info['row_count']} | {info['timestamp_min']} | "
            f"{info['timestamp_max']} | {', '.join(info['source_splits']) or '(none)'} |"
        )

    if metadata["comparability_warning"]:
        lines.extend(
            [
                "",
                "## Benchmark Warning",
                "",
                f"**{metadata['comparability_warning']}**",
            ]
        )

    lines.extend(
        [
            "",
            "## Chronological Boundaries",
            "",
            (
                "- Train to validation: "
                f"{metadata['chronological_boundary_timestamps']['train_max_timestamp']} -> "
                f"{metadata['chronological_boundary_timestamps']['validation_min_timestamp']}"
            ),
            (
                "- Validation to final partition: "
                f"{metadata['chronological_boundary_timestamps']['validation_max_timestamp']} -> "
                f"{metadata['chronological_boundary_timestamps']['test_min_timestamp']}"
            ),
            (
                "Rows are ordered by timestamp, then stable source-row number, then impression "
                "ID. If a tied timestamp crosses a boundary, earlier source rows are assigned "
                "to the earlier partition."
            ),
            "",
            "## Boundary Checks",
            "",
            (
                f"- Training max timestamp not later than validation min timestamp: "
                f"{metadata['integrity_checks']['train_max_not_later_than_validation_min']}"
            ),
            (
                "- Validation max timestamp not later than final partition min timestamp: "
                f"{metadata['integrity_checks']['validation_max_not_later_than_test_min']}"
            ),
            "",
            "## Integrity Checks",
            "",
        ]
    )
    if protocol == "official_train_dev":
        lines.extend(
            [
                (
                    "- Observed official dev period starts after model-development max "
                    f"timestamp: {observed['test_min_not_earlier_than_model_development_max']}"
                ),
                (
                    "The official dev split is reported as observed in the local files; this "
                    "pipeline does not assume it occurs after all official train timestamps."
                ),
                "",
            ]
        )
    for key, value in metadata["integrity_checks"].items():
        if isinstance(value, bool):
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Referenced News Coverage",
            "",
            "| Partition | Missing history ID occurrences | Missing candidate ID occurrences |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("train", "validation", "test"):
        coverage = metadata["referenced_news"][name]
        lines.append(
            f"| {name} | {coverage['missing_history_id_occurrences']} | "
            f"{coverage['missing_candidate_id_occurrences']} |"
        )

    lines.extend(
        [
            "",
            "## Parquet Schema",
            "",
            "News table:",
            "",
            *[f"- `{name}`: {description}" for name, description in NEWS_SCHEMA_DESCRIPTION.items()],
            "",
            "Behavior tables:",
            "",
            *[
                f"- `{name}`: {description}"
                for name, description in BEHAVIOR_SCHEMA_DESCRIPTION.items()
            ],
            "",
        ]
    )
    return "\n".join(lines)


def _chronological_train_count(total_rows: int, train_fraction: float) -> int:
    if total_rows <= 1:
        return total_rows
    count = int(total_rows * train_fraction)
    return min(max(count, 1), total_rows - 1)


def _train_only_boundaries(
    total_rows: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int]:
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))
    if train_end <= 0 or validation_end <= train_end or validation_end >= total_rows:
        raise MindDataError(
            "Train-only chronological splitting requires enough rows to create non-empty "
            "train, validation, and internal test partitions."
        )
    return train_end, validation_end


def _validate_train_only_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = {
        "--train-ratio": train_ratio,
        "--validation-ratio": validation_ratio,
        "--test-ratio": test_ratio,
    }
    if any(not 0 < value < 1 for value in ratios.values()):
        raise MindDataError("Train-only ratios must each be greater than 0 and less than 1.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise MindDataError("Train-only ratios must sum to 1.0.")


def _behavior_sort_key(record: BehaviorRecord) -> tuple[datetime, int, str]:
    return (record.timestamp, record.source_row_number, record.impression_id)


def _dedupe_news(records: list[NewsRecord]) -> list[NewsRecord]:
    seen: set[str] = set()
    deduped: list[NewsRecord] = []
    for record in records:
        if record.news_id in seen:
            continue
        seen.add(record.news_id)
        deduped.append(record)
    return deduped


def _build_metadata(
    *,
    protocol: DataProtocol,
    train_fraction: float,
    requested_ratios: dict[str, float],
    partitions: dict[str, list[BehaviorRecord]],
    integrity: dict[str, Any],
    news_records: list[NewsRecord],
) -> dict[str, Any]:
    train_max = _timestamp_max(partitions["train"])
    validation_min = _timestamp_min(partitions["validation"])
    validation_max = _timestamp_max(partitions["validation"])
    model_development_max = _timestamp_max([*partitions["train"], *partitions["validation"]])
    test_min = _timestamp_min(partitions["test"])
    news_ids = {record.news_id for record in news_records}
    row_counts = {name: len(rows) for name, rows in partitions.items()}
    total_rows = sum(row_counts.values())
    if protocol == "official_train_dev":
        metadata_ratios: dict[str, float | None] = {
            "train": train_fraction,
            "validation": round(1 - train_fraction, 10),
            "test": None,
        }
        final_partition_type = "official_dev_test"
        source_splits_used = ["train", "dev"]
        comparability_warning = None
    else:
        metadata_ratios = requested_ratios
        final_partition_type = "internal_chronological_holdout"
        source_splits_used = ["train"]
        comparability_warning = INTERNAL_HOLDOUT_WARNING

    return {
        "protocol": protocol,
        "source_splits_used": source_splits_used,
        "requested_ratios": metadata_ratios,
        "observed_row_counts": row_counts,
        "observed_ratios": {
            name: round(count / total_rows, 10) if total_rows else 0.0
            for name, count in row_counts.items()
        },
        "final_partition_type": final_partition_type,
        "comparability_warning": comparability_warning,
        "split_logic": {
            "official_train_source": "MINDsmall_train/behaviors.tsv",
            "official_dev_source": (
                "MINDsmall_dev/behaviors.tsv" if protocol == "official_train_dev" else None
            ),
            "train_fraction_within_official_train": (
                train_fraction if protocol == "official_train_dev" else None
            ),
            "validation_fraction_within_official_train": (
                round(1 - train_fraction, 10) if protocol == "official_train_dev" else None
            ),
            "train_only_ratios": (
                requested_ratios if protocol == "train_only_chronological" else None
            ),
            "sort_order": ["timestamp", "source_row_number", "impression_id"],
            "boundary_assignment": (
                "Partition boundaries are applied after stable chronological sorting. "
                "When timestamps tie across a boundary, lower source-row numbers are assigned "
                "to the earlier partition."
            ),
            "random_splitting_used": False,
            "history_policy": (
                "Histories are copied only from each source row's history field; no future "
                "interactions are used to reconstruct earlier examples."
            ),
        },
        "partitions": {
            name: _partition_summary(rows) for name, rows in partitions.items()
        },
        "observed_boundaries": {
            "train_max_timestamp": _iso_or_none(train_max),
            "validation_min_timestamp": _iso_or_none(validation_min),
            "model_development_max_timestamp": _iso_or_none(model_development_max),
            "test_min_timestamp": _iso_or_none(test_min),
            "test_min_not_earlier_than_model_development_max": _compare_timestamps(
                test_min,
                model_development_max,
            ),
        },
        "chronological_boundary_timestamps": {
            "train_max_timestamp": _iso_or_none(train_max),
            "validation_min_timestamp": _iso_or_none(validation_min),
            "validation_max_timestamp": _iso_or_none(validation_max),
            "test_min_timestamp": _iso_or_none(test_min),
        },
        "news": {
            "unique_news_count": len(news_records),
            "source_splits": sorted({record.source_split for record in news_records}),
        },
        "referenced_news": {
            name: _referenced_news_summary(rows, news_ids) for name, rows in partitions.items()
        },
        "integrity_checks": integrity,
        "schema": {
            "news.parquet": NEWS_SCHEMA_DESCRIPTION,
            "train_behaviors.parquet": BEHAVIOR_SCHEMA_DESCRIPTION,
            "validation_behaviors.parquet": BEHAVIOR_SCHEMA_DESCRIPTION,
            "test_behaviors.parquet": BEHAVIOR_SCHEMA_DESCRIPTION,
        },
    }


def _partition_summary(rows: list[BehaviorRecord]) -> dict[str, Any]:
    label_counts = Counter(
        candidate.clicked
        for row in rows
        for candidate in row.impressions
        if candidate.clicked is not None
    )
    return {
        "row_count": len(rows),
        "timestamp_min": _iso_or_none(_timestamp_min(rows)),
        "timestamp_max": _iso_or_none(_timestamp_max(rows)),
        "source_splits": sorted({row.source_split for row in rows}),
        "impression_id_count": len({row.impression_id for row in rows}),
        "clicked_candidate_count": label_counts.get(1, 0),
        "non_clicked_candidate_count": label_counts.get(0, 0),
        "unlabeled_candidate_count": sum(
            1 for row in rows for candidate in row.impressions if candidate.clicked is None
        ),
    }


def _integrity_checks(
    partitions: dict[str, list[BehaviorRecord]],
    news_ids: set[str],
    *,
    protocol: DataProtocol,
) -> dict[str, Any]:
    impression_overlaps = _partition_overlaps(
        {name: {row.impression_id for row in rows} for name, rows in partitions.items()}
    )
    source_row_overlaps = _partition_overlaps(
        {
            name: {f"{row.source_split}:{row.source_row_number}" for row in rows}
            for name, rows in partitions.items()
        }
    )
    row_hash_overlaps = _partition_overlaps(
        {name: {row.source_row_hash for row in rows} for name, rows in partitions.items()}
    )
    train_max = _timestamp_max(partitions["train"])
    validation_min = _timestamp_min(partitions["validation"])
    validation_max = _timestamp_max(partitions["validation"])
    test_min = _timestamp_min(partitions["test"])

    referenced_news = {
        name: _referenced_news_summary(rows, news_ids) for name, rows in partitions.items()
    }

    return {
        "no_impression_id_overlap": not any(impression_overlaps.values()),
        "impression_id_overlaps": impression_overlaps,
        "no_source_row_overlap": not any(source_row_overlaps.values()),
        "source_row_overlaps": source_row_overlaps,
        "no_identical_source_behavior_row_overlap": not any(row_hash_overlaps.values()),
        "source_behavior_row_hash_overlaps": row_hash_overlaps,
        "chronological_ordering_within_partitions": {
            name: _is_chronological(rows) for name, rows in partitions.items()
        },
        "all_partitions_chronological": all(
            _is_chronological(rows) for rows in partitions.values()
        ),
        "train_max_not_later_than_validation_min": _train_not_later_than_validation(
            train_max,
            validation_min,
        ),
        "validation_max_not_later_than_test_min": _train_not_later_than_validation(
            validation_max,
            test_min,
        ),
        "official_dev_rows_never_in_train_or_validation": all(
            row.source_split != "dev" for row in [*partitions["train"], *partitions["validation"]]
        ),
        "final_test_events_only_from_official_dev": (
            all(row.source_split == "dev" for row in partitions["test"])
            if protocol == "official_train_dev"
            else None
        ),
        "final_test_events_only_from_official_train": (
            all(row.source_split == "train" for row in partitions["test"])
            if protocol == "train_only_chronological"
            else None
        ),
        "random_splitting_used": False,
        "candidate_labels_preserved": all(
            candidate.clicked in {None, 0, 1}
            for rows in partitions.values()
            for row in rows
            for candidate in row.impressions
        ),
        "history_ordering_preserved": all(
            row.history == (row.history_raw.split() if row.history_raw else [])
            for rows in partitions.values()
            for row in rows
        ),
        "referenced_news": referenced_news,
    }


def _raise_for_failed_integrity(
    integrity: dict[str, Any],
    *,
    protocol: DataProtocol,
) -> None:
    required_checks = {
        "no_impression_id_overlap": integrity["no_impression_id_overlap"],
        "no_source_row_overlap": integrity["no_source_row_overlap"],
        "all_partitions_chronological": integrity["all_partitions_chronological"],
        "train_max_not_later_than_validation_min": integrity[
            "train_max_not_later_than_validation_min"
        ],
        "official_dev_rows_never_in_train_or_validation": integrity[
            "official_dev_rows_never_in_train_or_validation"
        ],
        "candidate_labels_preserved": integrity["candidate_labels_preserved"],
        "history_ordering_preserved": integrity["history_ordering_preserved"],
    }
    if protocol == "official_train_dev":
        required_checks["final_test_events_only_from_official_dev"] = integrity[
            "final_test_events_only_from_official_dev"
        ]
    else:
        required_checks["validation_max_not_later_than_test_min"] = integrity[
            "validation_max_not_later_than_test_min"
        ]
        required_checks["final_test_events_only_from_official_train"] = integrity[
            "final_test_events_only_from_official_train"
        ]
    failed = [name for name, passed in required_checks.items() if not passed]
    if failed:
        raise MindDataError(f"Integrity checks failed: {', '.join(failed)}")


def _partition_overlaps(partitions: dict[str, set[str]]) -> dict[str, list[str]]:
    overlaps: dict[str, list[str]] = {}
    for left, right in combinations(sorted(partitions), 2):
        overlaps[f"{left}__{right}"] = sorted(partitions[left] & partitions[right])
    return overlaps


def _is_chronological(rows: list[BehaviorRecord]) -> bool:
    return rows == sorted(rows, key=_behavior_sort_key)


def _train_not_later_than_validation(
    train_max: datetime | None,
    validation_min: datetime | None,
) -> bool:
    if train_max is None or validation_min is None:
        return True
    return train_max <= validation_min


def _compare_timestamps(left: datetime | None, right: datetime | None) -> bool | None:
    if left is None or right is None:
        return None
    return left >= right


def _display_ratio(value: float | None) -> str:
    return "source-defined" if value is None else f"{value:.2f}"


def _referenced_news_summary(rows: list[BehaviorRecord], news_ids: set[str]) -> dict[str, Any]:
    history_missing = [
        news_id for row in rows for news_id in row.history if news_id not in news_ids
    ]
    candidate_missing = [
        candidate.news_id
        for row in rows
        for candidate in row.impressions
        if candidate.news_id not in news_ids
    ]
    return {
        "missing_history_id_occurrences": len(history_missing),
        "missing_history_id_unique_count": len(set(history_missing)),
        "missing_history_id_samples": sorted(set(history_missing))[:20],
        "missing_candidate_id_occurrences": len(candidate_missing),
        "missing_candidate_id_unique_count": len(set(candidate_missing)),
        "missing_candidate_id_samples": sorted(set(candidate_missing))[:20],
    }


def _timestamp_min(rows: list[BehaviorRecord]) -> datetime | None:
    if not rows:
        return None
    return min(row.timestamp for row in rows)


def _timestamp_max(rows: list[BehaviorRecord]) -> datetime | None:
    if not rows:
        return None
    return max(row.timestamp for row in rows)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _write_news_parquet(records: list[NewsRecord], path: Path) -> None:
    table = pa.table(
        {
            "source_split": pa.array([record.source_split for record in records], pa.string()),
            "source_row_number": pa.array(
                [record.source_row_number for record in records],
                pa.int64(),
            ),
            "news_id": pa.array([record.news_id for record in records], pa.string()),
            "category": pa.array([record.category for record in records], pa.string()),
            "subcategory": pa.array([record.subcategory for record in records], pa.string()),
            "title": pa.array([record.title for record in records], pa.string()),
            "abstract": pa.array([record.abstract for record in records], pa.string()),
            "url": pa.array([record.url for record in records], pa.string()),
            "title_entities": pa.array([record.title_entities for record in records], pa.string()),
            "abstract_entities": pa.array(
                [record.abstract_entities for record in records],
                pa.string(),
            ),
        }
    )
    pq.write_table(table, path)


def _write_behaviors_parquet(records: list[BehaviorRecord], path: Path) -> None:
    impression_type = pa.list_(
        pa.struct(
            [
                pa.field("position", pa.int32(), nullable=False),
                pa.field("news_id", pa.string(), nullable=False),
                pa.field("clicked", pa.int8(), nullable=True),
            ]
        )
    )
    table = pa.table(
        {
            "source_split": pa.array([record.source_split for record in records], pa.string()),
            "source_row_number": pa.array(
                [record.source_row_number for record in records],
                pa.int64(),
            ),
            "source_row_hash": pa.array([record.source_row_hash for record in records], pa.string()),
            "impression_id": pa.array([record.impression_id for record in records], pa.string()),
            "user_id": pa.array([record.user_id for record in records], pa.string()),
            "timestamp": pa.array(
                [record.timestamp for record in records],
                pa.timestamp("us", tz="UTC"),
            ),
            "history": pa.array([record.history for record in records], pa.list_(pa.string())),
            "impressions": pa.array(
                [
                    [
                        {
                            "position": position,
                            "news_id": candidate.news_id,
                            "clicked": candidate.clicked,
                        }
                        for position, candidate in enumerate(record.impressions)
                    ]
                    for record in records
                ],
                impression_type,
            ),
        }
    )
    pq.write_table(table, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic MIND-small temporal splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--protocol",
        choices=DATA_PROTOCOLS,
        default=DEFAULT_DATA_PROTOCOL,
        help="Source and split protocol.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Official protocol fraction of train-source rows assigned to model train.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_ONLY_TRAIN_RATIO,
        help="Train-only protocol ratio assigned to train.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_TRAIN_ONLY_VALIDATION_RATIO,
        help="Train-only protocol ratio assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TRAIN_ONLY_TEST_RATIO,
        help="Train-only protocol ratio assigned to the internal holdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
            protocol=args.protocol,
            train_fraction=args.train_fraction,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
    except (MindDataError, FileNotFoundError) as exc:
        raise SystemExit(f"Preparation failed: {exc}") from exc

    print(f"Wrote {args.output_dir / 'news.parquet'}")
    print(f"Wrote {args.output_dir / 'train_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'validation_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'test_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'split_metadata.json'}")
    print(f"Wrote {args.reports_dir / 'split_summary.md'}")
    if args.protocol == "train_only_chronological":
        print(INTERNAL_HOLDOUT_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
