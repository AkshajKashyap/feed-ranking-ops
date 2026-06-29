from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import ValidationError

from feed_ranking_ops.serving.schemas import (
    PolicyManifest,
    RankRequest,
    RankedCandidate,
    RankResponse,
)

SUPPORTED_POLICY_FAMILIES = {"category_affinity", "original_order"}


class PolicyLoadError(RuntimeError):
    """Raised when a serving policy artifact is missing or inconsistent."""


@dataclass(frozen=True)
class ServingNewsItem:
    news_id: str
    category: str
    subcategory: str


@dataclass(frozen=True)
class CandidatePolicyScore:
    original_position: int
    news_id: str
    score: float


@dataclass(frozen=True)
class PolicyScoreResult:
    candidates: list[CandidatePolicyScore]
    missing_candidate_ids: list[str]
    unknown_history_ids: list[str]


@dataclass(frozen=True)
class PolicyRuntime:
    manifest: PolicyManifest
    news: dict[str, ServingNewsItem]

    def rank(self, request: RankRequest) -> RankResponse:
        result = self.score_candidates(
            history_news_ids=request.history_news_ids,
            candidate_news_ids=request.candidate_news_ids,
        )
        return self._response(request, result=result)

    def score_candidates(
        self,
        *,
        history_news_ids: list[str],
        candidate_news_ids: list[str],
    ) -> PolicyScoreResult:
        if self.manifest.selected_policy_family == "category_affinity":
            return self._score_category_affinity(
                history_news_ids,
                candidate_news_ids,
            )
        if self.manifest.selected_policy_family == "original_order":
            return self._score_original_order(
                history_news_ids,
                candidate_news_ids,
            )
        raise PolicyLoadError(
            f"Unsupported serving policy family: "
            f"{self.manifest.selected_policy_family}"
        )

    def _score_category_affinity(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str],
    ) -> PolicyScoreResult:
        category_counts: Counter[str] = Counter()
        subcategory_counts: Counter[str] = Counter()
        unknown_history_ids: list[str] = []
        for news_id in history_news_ids:
            item = self.news.get(news_id)
            if item is None:
                unknown_history_ids.append(news_id)
                continue
            category_counts[item.category] += 1
            subcategory_counts[item.subcategory] += 1

        category_weight = float(
            self.manifest.policy_config.get("category_weight", 1.0)
        )
        subcategory_weight = float(
            self.manifest.policy_config.get("subcategory_weight", 0.5)
        )
        fallback_score = float(
            self.manifest.policy_config.get("fallback_score", 0.0)
        )
        scored: list[CandidatePolicyScore] = []
        missing_candidate_ids: list[str] = []
        for position, news_id in enumerate(candidate_news_ids):
            item = self.news.get(news_id)
            if item is None:
                missing_candidate_ids.append(news_id)
                continue
            if category_counts or subcategory_counts:
                score = (
                    category_weight * category_counts[item.category]
                    + subcategory_weight * subcategory_counts[item.subcategory]
                )
            else:
                score = fallback_score
            scored.append(
                CandidatePolicyScore(
                    original_position=position,
                    news_id=news_id,
                    score=float(score),
                )
            )
        return PolicyScoreResult(
            candidates=scored,
            missing_candidate_ids=missing_candidate_ids,
            unknown_history_ids=_ordered_unique(unknown_history_ids),
        )

    def _score_original_order(
        self,
        history_news_ids: list[str],
        candidate_news_ids: list[str],
    ) -> PolicyScoreResult:
        unknown_history_ids = _ordered_unknown(history_news_ids, self.news)
        scored: list[CandidatePolicyScore] = []
        missing_candidate_ids: list[str] = []
        for position, news_id in enumerate(candidate_news_ids):
            if news_id not in self.news:
                missing_candidate_ids.append(news_id)
                continue
            scored.append(
                CandidatePolicyScore(
                    original_position=position,
                    news_id=news_id,
                    score=-float(position),
                )
            )
        return PolicyScoreResult(
            candidates=scored,
            missing_candidate_ids=missing_candidate_ids,
            unknown_history_ids=unknown_history_ids,
        )

    def _response(
        self,
        request: RankRequest,
        *,
        result: PolicyScoreResult,
    ) -> RankResponse:
        ordered = sorted(
            result.candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.original_position,
            ),
        )
        ranked = [
            RankedCandidate(
                news_id=candidate.news_id,
                original_position=candidate.original_position,
                rank=rank,
                score=candidate.score,
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]
        warnings: list[str] = []
        if not request.history_news_ids:
            warnings.append(
                "History is empty; candidates use fallback scores and source-order ties."
            )
        if result.unknown_history_ids:
            warnings.append("Unknown history IDs were excluded from the user profile.")
        if result.missing_candidate_ids:
            warnings.append("Unknown candidate IDs were omitted from ranked candidates.")
        if self.manifest.internal_holdout_warning:
            warnings.append(self.manifest.internal_holdout_warning)
        return RankResponse(
            selected_policy=self.manifest.selected_policy_name,
            policy_family=self.manifest.selected_policy_family,
            ranked_candidates=ranked,
            missing_candidate_ids=_ordered_unique(result.missing_candidate_ids),
            unknown_history_ids=result.unknown_history_ids,
            metadata={
                "requested_candidate_count": len(request.candidate_news_ids),
                "ranked_candidate_count": len(ranked),
                "known_history_count": (
                    len(request.history_news_ids)
                    - len(result.unknown_history_ids)
                ),
                "timestamp": (
                    request.timestamp.isoformat() if request.timestamp else None
                ),
                "tie_breaker": "original_candidate_position",
            },
            warnings=warnings,
        )


def load_policy_runtime(manifest_path: Path) -> PolicyRuntime:
    if not manifest_path.is_file():
        raise PolicyLoadError("Policy manifest is not configured or does not exist")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PolicyManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise PolicyLoadError("Policy manifest is malformed") from exc
    if not manifest.serving_ready:
        raise PolicyLoadError(
            "Selected policy has no supported serving artifact"
        )
    if manifest.selected_policy_family not in SUPPORTED_POLICY_FAMILIES:
        raise PolicyLoadError(
            f"Unsupported serving policy family: "
            f"{manifest.selected_policy_family}"
        )
    catalog_value = manifest.artifact_paths.get("news_catalog")
    if not catalog_value:
        raise PolicyLoadError("Policy manifest does not declare a news catalog artifact")
    catalog_relative = Path(catalog_value)
    if catalog_relative.is_absolute() or ".." in catalog_relative.parts:
        raise PolicyLoadError("News catalog path must stay within the serving artifact")
    catalog_path = manifest_path.parent / catalog_relative
    if not catalog_path.is_file():
        raise PolicyLoadError("News catalog artifact is missing")
    table = pq.read_table(catalog_path)
    required = {"news_id", "category", "subcategory"}
    if set(manifest.news_catalog_columns) != required:
        raise PolicyLoadError(
            "Policy manifest news catalog schema does not match the runtime contract"
        )
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise PolicyLoadError(
            f"News catalog is missing required columns: {', '.join(missing)}"
        )
    news: dict[str, ServingNewsItem] = {}
    for row in table.select(["news_id", "category", "subcategory"]).to_pylist():
        news_id = row["news_id"]
        if not isinstance(news_id, str) or not news_id:
            raise PolicyLoadError("News catalog contains an invalid news ID")
        if news_id in news:
            raise PolicyLoadError(f"News catalog contains duplicate ID {news_id!r}")
        news[news_id] = ServingNewsItem(
            news_id=news_id,
            category=_string(row["category"]),
            subcategory=_string(row["subcategory"]),
        )
    return PolicyRuntime(manifest=manifest, news=news)


def public_policy_metadata(manifest: PolicyManifest) -> dict[str, Any]:
    return {
        field: getattr(manifest, field)
        for field in (
            "selected_policy_name",
            "selected_policy_family",
            "selected_metric",
            "validation_metrics",
            "internal_test_metrics",
            "promotion_decision",
            "learned_promotion_result",
            "promotion_threshold_relative",
            "observed_validation_improvement_relative",
            "data_protocol",
            "final_partition_type",
            "internal_holdout_warning",
            "fitting_partitions",
            "serving_ready",
            "limitations",
        )
    }


def _ordered_unknown(
    values: list[str],
    known: dict[str, ServingNewsItem],
) -> list[str]:
    return _ordered_unique([value for value in values if value not in known])


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""
