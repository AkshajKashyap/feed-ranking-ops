from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from feed_ranking_ops.data.schemas import (
    BehaviorParseStats,
    BehaviorRecord,
    CandidateRecord,
    MindDataError,
    MindParseError,
    NewsParseStats,
    NewsRecord,
)

NEWS_COLUMN_COUNT = 8
BEHAVIOR_COLUMN_COUNT = 5

MIND_TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"
EXTRA_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
)


def parse_news_file(path: Path, split: str) -> tuple[list[NewsRecord], NewsParseStats]:
    """Parse a MIND news.tsv file with exact column-count validation."""
    if not path.exists():
        raise MindDataError(f"Missing news source file: {path}")

    records: list[NewsRecord] = []
    stats = NewsParseStats(split=split)

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")
            columns = line.split("\t")
            if len(columns) != NEWS_COLUMN_COUNT:
                raise MindParseError(
                    f"{path}:{line_number} expected {NEWS_COLUMN_COUNT} tab-separated "
                    f"columns for news.tsv, found {len(columns)}."
                )

            stats.row_count += 1
            (
                news_id,
                category,
                subcategory,
                title,
                abstract,
                url,
                title_entities,
                abstract_entities,
            ) = columns

            news_id = news_id.strip()
            if not news_id:
                raise MindParseError(f"{path}:{line_number} news_id is required.")

            records.append(
                NewsRecord(
                    source_split=split,
                    source_row_number=line_number,
                    news_id=news_id,
                    category=category.strip(),
                    subcategory=subcategory.strip(),
                    title=title.strip(),
                    abstract=abstract.strip(),
                    url=url.strip(),
                    title_entities=title_entities.strip(),
                    abstract_entities=abstract_entities.strip(),
                )
            )

    return records, stats


def parse_behavior_file(
    path: Path,
    split: str,
    *,
    strict: bool = True,
) -> tuple[list[BehaviorRecord], BehaviorParseStats]:
    """Parse a MIND behaviors.tsv file.

    In strict mode, invalid timestamps and malformed impression tokens raise
    MindParseError. In non-strict mode, invalid timestamp rows are omitted from
    returned records and malformed impression tokens are skipped; both cases are
    counted in the returned stats for audit reporting.
    """
    if not path.exists():
        raise MindDataError(f"Missing behavior source file: {path}")

    records: list[BehaviorRecord] = []
    stats = BehaviorParseStats(split=split)

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")
            columns = line.split("\t")
            if len(columns) != BEHAVIOR_COLUMN_COUNT:
                raise MindParseError(
                    f"{path}:{line_number} expected {BEHAVIOR_COLUMN_COUNT} tab-separated "
                    f"columns for behaviors.tsv, found {len(columns)}."
                )

            stats.row_count += 1
            impression_id, user_id, timestamp_text, history_text, impressions_text = columns
            impression_id = impression_id.strip()
            user_id = user_id.strip()
            timestamp_text = timestamp_text.strip()
            history_text = history_text.strip()
            impressions_text = impressions_text.strip()

            if not impression_id:
                raise MindParseError(f"{path}:{line_number} impression_id is required.")
            if not user_id:
                raise MindParseError(f"{path}:{line_number} user_id is required.")

            stats.impression_ids.add(impression_id)
            stats.user_ids.add(user_id)

            history = history_text.split() if history_text else []
            if not history:
                stats.empty_history_count += 1
            stats.history_lengths.append(len(history))

            impressions = _parse_impressions(
                impressions_text=impressions_text,
                path=path,
                line_number=line_number,
                strict=strict,
                stats=stats,
            )
            stats.candidate_counts.append(len(impressions))

            try:
                timestamp = parse_timestamp(timestamp_text)
            except ValueError as exc:
                stats.invalid_timestamp_count += 1
                if strict:
                    raise MindParseError(
                        f"{path}:{line_number} invalid timestamp {timestamp_text!r}. "
                        f"Expected MIND format like '11/15/2019 8:00:00 AM' or ISO-8601."
                    ) from exc
                continue

            records.append(
                BehaviorRecord(
                    source_split=split,
                    source_row_number=line_number,
                    source_row_hash=hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    impression_id=impression_id,
                    user_id=user_id,
                    timestamp=timestamp,
                    history=history,
                    history_raw=history_text,
                    impressions=impressions,
                )
            )

    return records, stats


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")

    for timestamp_format in (MIND_TIMESTAMP_FORMAT, *EXTRA_TIMESTAMP_FORMATS):
        try:
            return _normalize_datetime(datetime.strptime(text, timestamp_format))
        except ValueError:
            pass

    try:
        return _normalize_datetime(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp: {value!r}") from exc


def parse_impression_token(token: str) -> CandidateRecord:
    token = token.strip()
    if not token:
        raise ValueError("empty impression token")

    if "-" not in token:
        return CandidateRecord(news_id=token, clicked=None)

    news_id, label = token.rsplit("-", 1)
    if not news_id or label not in {"0", "1"}:
        raise ValueError(
            "expected impression token in NEWS_ID, NEWS_ID-0, or NEWS_ID-1 format"
        )

    return CandidateRecord(news_id=news_id, clicked=int(label))


def _parse_impressions(
    *,
    impressions_text: str,
    path: Path,
    line_number: int,
    strict: bool,
    stats: BehaviorParseStats,
) -> list[CandidateRecord]:
    impressions: list[CandidateRecord] = []
    tokens = impressions_text.split() if impressions_text else []

    for token in tokens:
        try:
            candidate = parse_impression_token(token)
        except ValueError as exc:
            stats.malformed_impression_token_count += 1
            if strict:
                raise MindParseError(
                    f"{path}:{line_number} malformed impression token {token!r}. "
                    "Expected NEWS_ID, NEWS_ID-0, or NEWS_ID-1."
                ) from exc
            continue

        impressions.append(candidate)
        stats.candidate_news_ids.append(candidate.news_id)
        if candidate.clicked == 1:
            stats.clicked_candidate_count += 1
        elif candidate.clicked == 0:
            stats.non_clicked_candidate_count += 1
        else:
            stats.unlabeled_candidate_count += 1

    return impressions


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
