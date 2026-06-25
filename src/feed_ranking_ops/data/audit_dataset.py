from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from feed_ranking_ops.data.layout import EXPECTED_MIND_FILES, require_valid_mind_layout
from feed_ranking_ops.data.parsers import parse_behavior_file, parse_news_file
from feed_ranking_ops.data.schemas import (
    BehaviorParseStats,
    MindDataError,
    NewsParseStats,
    NewsRecord,
)


def audit_dataset(data_dir: Path, reports_dir: Path) -> dict[str, Any]:
    require_valid_mind_layout(data_dir)

    train_news, train_news_stats = parse_news_file(
        data_dir / EXPECTED_MIND_FILES["train_news"], "train"
    )
    dev_news, dev_news_stats = parse_news_file(data_dir / EXPECTED_MIND_FILES["dev_news"], "dev")
    train_behaviors, train_behavior_stats = parse_behavior_file(
        data_dir / EXPECTED_MIND_FILES["train_behaviors"], "train", strict=False
    )
    dev_behaviors, dev_behavior_stats = parse_behavior_file(
        data_dir / EXPECTED_MIND_FILES["dev_behaviors"], "dev", strict=False
    )

    news_records = [*train_news, *dev_news]
    news_stats = [train_news_stats, dev_news_stats]
    behavior_stats = [train_behavior_stats, dev_behavior_stats]
    all_candidate_news_ids = [
        news_id for stats in behavior_stats for news_id in stats.candidate_news_ids
    ]
    known_news_ids = {record.news_id for record in news_records}
    missing_candidate_ids = [
        news_id for news_id in all_candidate_news_ids if news_id not in known_news_ids
    ]

    audit = {
        "news": _build_news_audit(news_records, news_stats),
        "behaviors": _build_behavior_audit(behavior_stats, missing_candidate_ids),
        "parser_notes": {
            "timestamp_normalization": "Parsed timestamps are normalized to UTC.",
            "audit_mode": (
                "Rows with invalid timestamps are counted and omitted from parsed behavior "
                "records during audit. Malformed impression tokens are skipped and counted. "
                "The preparation command uses strict parsing and fails on either condition."
            ),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "data_audit.md").write_text(
        render_audit_markdown(audit),
        encoding="utf-8",
    )

    return audit


def render_audit_markdown(audit: dict[str, Any]) -> str:
    news = audit["news"]
    behaviors = audit["behaviors"]
    lines = [
        "# MIND-small Data Audit",
        "",
        "This report summarizes the local MIND-small files. No dataset download is performed.",
        "",
        "## News",
        "",
        f"- Total rows: {news['total_row_count']}",
        f"- Unique news IDs: {news['unique_news_count']}",
        f"- Duplicate news ID rows: {news['duplicate_news_id_count']}",
        f"- Missing titles: {news['missing_title_count']}",
        f"- Missing abstracts: {news['missing_abstract_count']}",
        "",
        "Rows by source split:",
        "",
        _markdown_table(["Split", "Rows"], news["row_count_by_source_split"].items()),
        "",
        "Top categories:",
        "",
        _markdown_table(["Category", "Rows"], news["category_counts"].items()),
        "",
        "Top subcategories:",
        "",
        _markdown_table(["Subcategory", "Rows"], news["subcategory_counts"].items()),
        "",
        _news_findings(news),
        "",
        "## Behaviors",
        "",
        f"- Total behavior rows: {behaviors['total_row_count']}",
        f"- Parsed behavior rows: {behaviors['parsed_row_count']}",
        f"- Unique users: {behaviors['unique_user_count']}",
        f"- Unique impressions: {behaviors['unique_impression_count']}",
        f"- Invalid timestamp rows: {behaviors['invalid_timestamp_count']}",
        f"- Empty histories: {behaviors['empty_history_count']}",
        f"- Malformed impression tokens: {behaviors['malformed_impression_token_count']}",
        f"- Average history length: {behaviors['average_history_length']}",
        f"- Median history length: {behaviors['median_history_length']}",
        f"- Average candidate count: {behaviors['average_candidate_count']}",
        f"- Median candidate count: {behaviors['median_candidate_count']}",
        f"- Clicked labeled candidates: {behaviors['clicked_candidate_count']}",
        f"- Non-clicked labeled candidates: {behaviors['non_clicked_candidate_count']}",
        f"- Unlabeled candidates: {behaviors['unlabeled_candidate_count']}",
        f"- Click-through rate: {behaviors['click_through_rate']}",
        "",
        "Rows by source split:",
        "",
        _markdown_table(["Split", "Rows"], behaviors["row_count_by_source_split"].items()),
        "",
        "Referenced candidate news IDs not found in news.tsv:",
        "",
        f"- Missing candidate occurrences: {behaviors['candidate_news_ids_not_found_count']}",
        f"- Unique missing candidate IDs: {behaviors['candidate_news_ids_not_found_unique_count']}",
        "",
        _behavior_findings(behaviors),
        "",
        "## Audit Semantics",
        "",
        (
            "Audit mode counts invalid timestamps and malformed impression tokens so data quality "
            "issues are visible. Preparation is stricter: these issues must be fixed before "
            "processed Parquet outputs are written."
        ),
        "",
    ]
    return "\n".join(lines)


def _build_news_audit(
    records: list[NewsRecord],
    stats_by_split: list[NewsParseStats],
) -> dict[str, Any]:
    news_id_counts = Counter(record.news_id for record in records)
    category_counts = Counter(_display_value(record.category) for record in records)
    subcategory_counts = Counter(_display_value(record.subcategory) for record in records)
    return {
        "row_count_by_source_split": {
            stats.split: stats.row_count for stats in sorted(stats_by_split, key=lambda s: s.split)
        },
        "total_row_count": len(records),
        "unique_news_count": len(news_id_counts),
        "duplicate_news_id_count": sum(count - 1 for count in news_id_counts.values() if count > 1),
        "missing_title_count": sum(1 for record in records if not record.title),
        "missing_abstract_count": sum(1 for record in records if not record.abstract),
        "category_counts": dict(category_counts.most_common()),
        "subcategory_counts": dict(subcategory_counts.most_common()),
    }


def _build_behavior_audit(
    stats_by_split: list[BehaviorParseStats],
    missing_candidate_ids: list[str],
) -> dict[str, Any]:
    history_lengths = [length for stats in stats_by_split for length in stats.history_lengths]
    candidate_counts = [count for stats in stats_by_split for count in stats.candidate_counts]
    clicked = sum(stats.clicked_candidate_count for stats in stats_by_split)
    non_clicked = sum(stats.non_clicked_candidate_count for stats in stats_by_split)
    missing_counter = Counter(missing_candidate_ids)

    return {
        "row_count_by_source_split": {
            stats.split: stats.row_count for stats in sorted(stats_by_split, key=lambda s: s.split)
        },
        "total_row_count": sum(stats.row_count for stats in stats_by_split),
        "parsed_row_count": sum(
            stats.row_count - stats.invalid_timestamp_count for stats in stats_by_split
        ),
        "unique_user_count": len({user for stats in stats_by_split for user in stats.user_ids}),
        "unique_impression_count": len(
            {impression for stats in stats_by_split for impression in stats.impression_ids}
        ),
        "invalid_timestamp_count": sum(stats.invalid_timestamp_count for stats in stats_by_split),
        "empty_history_count": sum(stats.empty_history_count for stats in stats_by_split),
        "average_history_length": _mean(history_lengths),
        "median_history_length": _median(history_lengths),
        "average_candidate_count": _mean(candidate_counts),
        "median_candidate_count": _median(candidate_counts),
        "clicked_candidate_count": clicked,
        "non_clicked_candidate_count": non_clicked,
        "unlabeled_candidate_count": sum(stats.unlabeled_candidate_count for stats in stats_by_split),
        "click_through_rate": _click_through_rate(clicked, non_clicked),
        "malformed_impression_token_count": sum(
            stats.malformed_impression_token_count for stats in stats_by_split
        ),
        "candidate_news_ids_not_found_count": len(missing_candidate_ids),
        "candidate_news_ids_not_found_unique_count": len(missing_counter),
        "candidate_news_ids_not_found_samples": sorted(missing_counter)[:20],
    }


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(mean(values), 4)


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _click_through_rate(clicked: int, non_clicked: int) -> float | None:
    labeled_total = clicked + non_clicked
    if labeled_total == 0:
        return None
    return round(clicked / labeled_total, 6)


def _display_value(value: str) -> str:
    return value if value else "(missing)"


def _markdown_table(headers: list[str], rows: Any) -> str:
    row_list = list(rows)
    if not row_list:
        return "_No rows._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for key, value in row_list:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _news_findings(news: dict[str, Any]) -> str:
    findings: list[str] = []
    if news["duplicate_news_id_count"]:
        findings.append(
            "Duplicate news IDs are present. Preparation keeps the first occurrence for "
            "the unified news table and records duplicate counts in the audit."
        )
    if news["missing_abstract_count"]:
        findings.append(
            "Some news rows have missing abstracts. This is allowed because MIND-small "
            "contains blank abstract fields, but downstream text features should account for it."
        )
    if news["missing_title_count"]:
        findings.append("Some news rows have missing titles and should be reviewed before modeling.")
    if not findings:
        findings.append("No duplicate news IDs or missing required news text fields were observed.")
    return "\n".join(f"- {finding}" for finding in findings)


def _behavior_findings(behaviors: dict[str, Any]) -> str:
    findings: list[str] = []
    if behaviors["invalid_timestamp_count"]:
        findings.append("Invalid timestamp rows were found; preparation will fail until fixed.")
    if behaviors["malformed_impression_token_count"]:
        findings.append("Malformed impression tokens were found and skipped in audit-mode counts.")
    if behaviors["candidate_news_ids_not_found_count"]:
        findings.append(
            "Some candidate IDs are absent from news.tsv. The IDs are measured here so the "
            "coverage gap is explicit before feature generation."
        )
    if behaviors["unlabeled_candidate_count"]:
        findings.append(
            "Unlabeled impression candidates are present. They are preserved with null click labels."
        )
    if not findings:
        findings.append("No blocking behavior parsing issues were observed.")
    return "\n".join(f"- {finding}" for finding in findings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local MIND-small source files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_dataset(args.data_dir, args.reports_dir)
    except (MindDataError, FileNotFoundError) as exc:
        raise SystemExit(f"Audit failed: {exc}") from exc

    print(f"Wrote {args.reports_dir / 'data_audit.json'}")
    print(f"Wrote {args.reports_dir / 'data_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
