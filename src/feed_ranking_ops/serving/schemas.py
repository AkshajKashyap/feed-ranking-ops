from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PolicyManifest(BaseModel):
    schema_version: int = 1
    request_schema_version: int = 1
    news_catalog_columns: list[str]
    selected_policy_name: str
    selected_policy_family: str
    selected_metric: str
    validation_metrics: dict[str, float | None]
    internal_test_metrics: dict[str, float | None]
    promotion_decision: Literal[
        "promote_learned_ranker",
        "promote_baseline_policy",
    ]
    learned_promotion_result: Literal[
        "promoted",
        "rejected_insufficient_improvement",
    ]
    promotion_threshold_relative: float = Field(ge=0.0)
    observed_validation_improvement_relative: float
    created_at: str
    git_commit: str | None
    data_protocol: str
    final_partition_type: str
    internal_holdout_warning: str | None
    fitting_partitions: list[str]
    policy_config: dict[str, Any]
    artifact_paths: dict[str, str]
    serving_ready: bool
    limitations: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    policy_loaded: bool
    selected_policy: str | None = None
    error: str | None = None


class PolicyResponse(BaseModel):
    selected_policy_name: str
    selected_policy_family: str
    selected_metric: str
    validation_metrics: dict[str, float | None]
    internal_test_metrics: dict[str, float | None]
    promotion_decision: str
    learned_promotion_result: str
    promotion_threshold_relative: float
    observed_validation_improvement_relative: float
    data_protocol: str
    final_partition_type: str
    internal_holdout_warning: str | None
    fitting_partitions: list[str]
    serving_ready: bool
    limitations: list[str]


class RankRequest(BaseModel):
    history_news_ids: list[str] = Field(default_factory=list, max_length=1000)
    candidate_news_ids: list[str] = Field(min_length=1, max_length=1000)
    timestamp: datetime | None = None

    @field_validator("history_news_ids", "candidate_news_ids")
    @classmethod
    def validate_news_ids(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("news IDs must be non-empty strings")
            cleaned.append(value.strip())
        return cleaned


class RankedCandidate(BaseModel):
    news_id: str
    original_position: int
    rank: int
    score: float


class RankResponse(BaseModel):
    selected_policy: str
    policy_family: str
    ranked_candidates: list[RankedCandidate]
    missing_candidate_ids: list[str]
    unknown_history_ids: list[str]
    metadata: dict[str, Any]
    warnings: list[str]


class RankRequestLogEvent(BaseModel):
    timestamp: str
    request_id: str
    selected_policy: str | None
    history_id_count: int = Field(ge=0)
    candidate_id_count: int = Field(ge=0)
    ranked_candidate_count: int = Field(ge=0)
    missing_candidate_count: int = Field(ge=0)
    unknown_history_count: int = Field(ge=0)
    empty_history: bool
    latency_ms: float = Field(ge=0.0)
    warnings: list[str]
    status: Literal["success", "failed"]
    outcome: str
    top_ranked_category_counts: dict[str, int]


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    missing_candidate_rate: float
    unknown_history_rate: float
    empty_history_rate: float
    request_logging_enabled: bool
    request_log_write_errors: int
