from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    NewsItem,
)


@dataclass(frozen=True)
class CandidateRow:
    partition: str
    impression_id: str
    user_id: str
    timestamp: datetime
    candidate_position: int
    candidate_news_id: str
    click_label: int | None
    history_news_ids: list[str]
    candidate_category: str
    candidate_subcategory: str
    candidate_title: str
    candidate_abstract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "impression_id": self.impression_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "candidate_position": self.candidate_position,
            "candidate_news_id": self.candidate_news_id,
            "click_label": self.click_label,
            "history_news_ids": list(self.history_news_ids),
            "candidate_category": self.candidate_category,
            "candidate_subcategory": self.candidate_subcategory,
            "candidate_title": self.candidate_title,
            "candidate_abstract": self.candidate_abstract,
        }


def behavior_to_candidate_rows(
    behavior: BehaviorImpression,
    news: dict[str, NewsItem],
) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    for candidate in behavior.candidates:
        item = news[candidate.news_id]
        rows.append(
            CandidateRow(
                partition=behavior.partition,
                impression_id=behavior.impression_id,
                user_id=behavior.user_id,
                timestamp=behavior.timestamp,
                candidate_position=candidate.position,
                candidate_news_id=candidate.news_id,
                click_label=candidate.clicked,
                history_news_ids=list(behavior.history_news_ids),
                candidate_category=item.category,
                candidate_subcategory=item.subcategory,
                candidate_title=item.title,
                candidate_abstract=item.abstract,
            )
        )
    return rows


def behaviors_to_candidate_rows(
    behaviors: list[BehaviorImpression],
    news: dict[str, NewsItem],
) -> list[CandidateRow]:
    return [
        row
        for behavior in behaviors
        for row in behavior_to_candidate_rows(behavior, news)
    ]


def candidate_rows_to_pyarrow(rows: list[CandidateRow]) -> pa.Table:
    payload = [row.to_dict() for row in rows]
    schema = pa.schema(
        [
            pa.field("partition", pa.string()),
            pa.field("impression_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("candidate_position", pa.int32()),
            pa.field("candidate_news_id", pa.string()),
            pa.field("click_label", pa.int8()),
            pa.field("history_news_ids", pa.list_(pa.string())),
            pa.field("candidate_category", pa.string()),
            pa.field("candidate_subcategory", pa.string()),
            pa.field("candidate_title", pa.string()),
            pa.field("candidate_abstract", pa.string()),
        ]
    )
    return pa.Table.from_pylist(payload, schema=schema)


def write_candidate_rows_parquet(rows: list[CandidateRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidate_rows_to_pyarrow(rows), path)
