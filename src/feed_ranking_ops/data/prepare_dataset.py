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

from feed_ranking_ops.data.layout import EXPECTED_MIND_FILES, require_valid_mind_layout
from feed_ranking_ops.data.parsers import parse_behavior_file, parse_news_file
from feed_ranking_ops.data.schemas import BehaviorRecord, MindDataError, NewsRecord

DEFAULT_TRAIN_FRACTION = 0.8

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
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, Any]:
    if not 0 < train_fraction < 1:
        raise MindDataError("--train-fraction must be greater than 0 and less than 1.")

    require_valid_mind_layout(data_dir)

    train_news, _ = parse_news_file(data_dir / EXPECTED_MIND_FILES["train_news"], "train")
    dev_news, _ = parse_news_file(data_dir / EXPECTED_MIND_FILES["dev_news"], "dev")
    train_behaviors, _ = parse_behavior_file(
        data_dir / EXPECTED_MIND_FILES["train_behaviors"], "train", strict=True
    )
    dev_behaviors, _ = parse_behavior_file(
        data_dir / EXPECTED_MIND_FILES["dev_behaviors"], "dev", strict=True
    )

    train_sorted = sorted(train_behaviors, key=_behavior_sort_key)
    dev_sorted = sorted(dev_behaviors, key=_behavior_sort_key)
    train_count = _chronological_train_count(len(train_sorted), train_fraction)
    train_partition = train_sorted[:train_count]
    validation_partition = train_sorted[train_count:]
    test_partition = dev_sorted

    news_records = _dedupe_news([*train_news, *dev_news])
    news_ids = {record.news_id for record in news_records}
    partitions = {
        "train": train_partition,
        "validation": validation_partition,
        "test": test_partition,
    }
    integrity = _integrity_checks(partitions, news_ids)
    _raise_for_failed_integrity(integrity)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_news_parquet(news_records, output_dir / "news.parquet")
    _write_behaviors_parquet(train_partition, output_dir / "train_behaviors.parquet")
    _write_behaviors_parquet(validation_partition, output_dir / "validation_behaviors.parquet")
    _write_behaviors_parquet(test_partition, output_dir / "test_behaviors.parquet")

    metadata = _build_metadata(
        train_fraction=train_fraction,
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
    lines = [
        "# MIND-small Temporal Split Summary",
        "",
        "Official MIND train behaviors are sorted chronologically and split into model "
        "train and validation partitions. Official MIND dev behaviors are kept as the "
        "offline test partition.",
        "",
        "## Partition Counts",
        "",
        "| Partition | Rows | Timestamp min | Timestamp max | Source split |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name in ("train", "validation", "test"):
        info = partitions[name]
        lines.append(
            f"| {name} | {info['row_count']} | {info['timestamp_min']} | "
            f"{info['timestamp_max']} | {', '.join(info['source_splits']) or '(none)'} |"
        )

    lines.extend(
        [
            "",
            "## Boundary Checks",
            "",
            (
                f"- Training max timestamp not later than validation min timestamp: "
                f"{metadata['integrity_checks']['train_max_not_later_than_validation_min']}"
            ),
            (
                f"- Observed test period starts after model-development max timestamp: "
                f"{observed['test_min_not_earlier_than_model_development_max']}"
            ),
            (
                "The official dev split is reported as observed in the local files; this "
                "pipeline does not assume it occurs after all official train timestamps."
            ),
            "",
            "## Integrity Checks",
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
    train_fraction: float,
    partitions: dict[str, list[BehaviorRecord]],
    integrity: dict[str, Any],
    news_records: list[NewsRecord],
) -> dict[str, Any]:
    train_max = _timestamp_max(partitions["train"])
    validation_min = _timestamp_min(partitions["validation"])
    model_development_max = _timestamp_max([*partitions["train"], *partitions["validation"]])
    test_min = _timestamp_min(partitions["test"])
    news_ids = {record.news_id for record in news_records}

    return {
        "split_logic": {
            "official_train_source": "MINDsmall_train/behaviors.tsv",
            "official_dev_source": "MINDsmall_dev/behaviors.tsv",
            "train_fraction_within_official_train": train_fraction,
            "validation_fraction_within_official_train": round(1 - train_fraction, 10),
            "sort_order": ["timestamp", "source_row_number", "impression_id"],
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
) -> dict[str, Any]:
    impression_overlaps = _partition_overlaps(
        {name: {row.impression_id for row in rows} for name, rows in partitions.items()}
    )
    row_hash_overlaps = _partition_overlaps(
        {name: {row.source_row_hash for row in rows} for name, rows in partitions.items()}
    )
    train_max = _timestamp_max(partitions["train"])
    validation_min = _timestamp_min(partitions["validation"])

    referenced_news = {
        name: _referenced_news_summary(rows, news_ids) for name, rows in partitions.items()
    }

    return {
        "no_impression_id_overlap": not any(impression_overlaps.values()),
        "impression_id_overlaps": impression_overlaps,
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
        "official_dev_rows_never_in_train_or_validation": all(
            row.source_split != "dev" for row in [*partitions["train"], *partitions["validation"]]
        ),
        "final_test_events_only_from_official_dev": all(
            row.source_split == "dev" for row in partitions["test"]
        ),
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


def _raise_for_failed_integrity(integrity: dict[str, Any]) -> None:
    required_checks = {
        "no_impression_id_overlap": integrity["no_impression_id_overlap"],
        "no_identical_source_behavior_row_overlap": integrity[
            "no_identical_source_behavior_row_overlap"
        ],
        "all_partitions_chronological": integrity["all_partitions_chronological"],
        "train_max_not_later_than_validation_min": integrity[
            "train_max_not_later_than_validation_min"
        ],
        "official_dev_rows_never_in_train_or_validation": integrity[
            "official_dev_rows_never_in_train_or_validation"
        ],
        "final_test_events_only_from_official_dev": integrity[
            "final_test_events_only_from_official_dev"
        ],
        "candidate_labels_preserved": integrity["candidate_labels_preserved"],
        "history_ordering_preserved": integrity["history_ordering_preserved"],
    }
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
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Chronological fraction of official train behaviors assigned to train.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
            train_fraction=args.train_fraction,
        )
    except (MindDataError, FileNotFoundError) as exc:
        raise SystemExit(f"Preparation failed: {exc}") from exc

    print(f"Wrote {args.output_dir / 'news.parquet'}")
    print(f"Wrote {args.output_dir / 'train_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'validation_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'test_behaviors.parquet'}")
    print(f"Wrote {args.output_dir / 'split_metadata.json'}")
    print(f"Wrote {args.reports_dir / 'split_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
