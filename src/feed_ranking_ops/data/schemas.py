from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class MindDataError(Exception):
    """Base exception for actionable MIND data issues."""


class MindParseError(MindDataError):
    """Raised when a MIND TSV row cannot be parsed safely."""


@dataclass(frozen=True)
class CandidateRecord:
    news_id: str
    clicked: int | None


@dataclass(frozen=True)
class NewsRecord:
    source_split: str
    source_row_number: int
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str


@dataclass(frozen=True)
class BehaviorRecord:
    source_split: str
    source_row_number: int
    source_row_hash: str
    impression_id: str
    user_id: str
    timestamp: datetime
    history: list[str]
    history_raw: str
    impressions: list[CandidateRecord]


@dataclass
class NewsParseStats:
    split: str
    row_count: int = 0


@dataclass
class BehaviorParseStats:
    split: str
    row_count: int = 0
    invalid_timestamp_count: int = 0
    malformed_impression_token_count: int = 0
    empty_history_count: int = 0
    user_ids: set[str] = field(default_factory=set)
    impression_ids: set[str] = field(default_factory=set)
    history_lengths: list[int] = field(default_factory=list)
    candidate_counts: list[int] = field(default_factory=list)
    clicked_candidate_count: int = 0
    non_clicked_candidate_count: int = 0
    unlabeled_candidate_count: int = 0
    candidate_news_ids: list[str] = field(default_factory=list)
