from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from feed_ranking_ops.serving.policy import (
    PolicyLoadError,
    PolicyRuntime,
    load_policy_runtime,
    public_policy_metadata,
)
from feed_ranking_ops.serving.observability import ServingObservability
from feed_ranking_ops.serving.schemas import (
    HealthResponse,
    MetricsResponse,
    PolicyResponse,
    RankRequest,
    RankRequestLogEvent,
    RankResponse,
)

DEFAULT_MANIFEST_PATH = Path("artifacts/serving/policy_manifest.json")


def create_app(
    manifest_path: Path | None = None,
    *,
    request_log_path: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    request_id_factory: Callable[[], str] | None = None,
) -> FastAPI:
    configured_path = manifest_path
    event_clock = clock or (lambda: datetime.now(UTC))
    make_request_id = request_id_factory or (lambda: str(uuid4()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected_path = configured_path or Path(
            os.environ.get(
                "FEED_RANKING_POLICY_MANIFEST",
                str(DEFAULT_MANIFEST_PATH),
            )
        )
        app.state.policy_runtime = None
        app.state.policy_error = None
        configured_log = request_log_path
        if configured_log is None:
            environment_log = os.environ.get("FEED_RANKING_OPS_REQUEST_LOG")
            configured_log = Path(environment_log) if environment_log else None
        app.state.observability = ServingObservability(
            request_log_path=configured_log,
        )
        try:
            app.state.policy_runtime = load_policy_runtime(selected_path)
        except PolicyLoadError as exc:
            app.state.policy_error = str(exc)
        yield

    application = FastAPI(
        title="FeedRank Ops Logged-Candidate Ranking API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        runtime = _runtime(request)
        if runtime is None:
            return HealthResponse(
                status="degraded",
                policy_loaded=False,
                error=request.app.state.policy_error,
            )
        return HealthResponse(
            status="ok",
            policy_loaded=True,
            selected_policy=runtime.manifest.selected_policy_name,
        )

    @application.get("/policy", response_model=PolicyResponse)
    async def policy(request: Request) -> PolicyResponse:
        runtime = _require_runtime(request)
        return PolicyResponse.model_validate(
            public_policy_metadata(runtime.manifest)
        )

    @application.post("/rank", response_model=RankResponse)
    async def rank(payload: RankRequest, request: Request) -> RankResponse:
        started = perf_counter()
        runtime = _runtime(request)
        response: RankResponse | None = None
        status = "failed"
        outcome = "service_unavailable"
        try:
            runtime = _require_runtime(request)
            response = runtime.rank(payload)
            status = "success"
            outcome = "ranked"
            return response
        except HTTPException:
            raise
        except (PolicyLoadError, ValueError) as exc:
            outcome = type(exc).__name__
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            latency_ms = max((perf_counter() - started) * 1000.0, 0.0)
            selected_policy = (
                runtime.manifest.selected_policy_name
                if runtime is not None
                else None
            )
            event = RankRequestLogEvent(
                timestamp=_utc_timestamp(event_clock()),
                request_id=_request_id(request, make_request_id),
                selected_policy=selected_policy,
                history_id_count=len(payload.history_news_ids),
                candidate_id_count=len(payload.candidate_news_ids),
                ranked_candidate_count=(
                    len(response.ranked_candidates) if response else 0
                ),
                missing_candidate_count=(
                    len(response.missing_candidate_ids) if response else 0
                ),
                unknown_history_count=(
                    len(response.unknown_history_ids) if response else 0
                ),
                empty_history=not payload.history_news_ids,
                latency_ms=latency_ms,
                warnings=response.warnings if response else [],
                status=status,
                outcome=outcome,
                top_ranked_category_counts=(
                    _top_ranked_categories(runtime, response)
                    if runtime is not None and response is not None
                    else {}
                ),
            )
            _observability(request).record(event)

    @application.get("/metrics", response_model=MetricsResponse)
    async def metrics(request: Request) -> MetricsResponse:
        return MetricsResponse.model_validate(
            _observability(request).summary()
        )

    return application


def _runtime(request: Request) -> PolicyRuntime | None:
    return request.app.state.policy_runtime


def _require_runtime(request: Request) -> PolicyRuntime:
    runtime = _runtime(request)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.policy_error or "Serving policy is unavailable",
        )
    return runtime


def _observability(request: Request) -> ServingObservability:
    return request.app.state.observability


def _request_id(
    request: Request,
    factory: Callable[[], str],
) -> str:
    supplied = request.headers.get("x-request-id", "").strip()
    return supplied[:128] if supplied else factory()


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _top_ranked_categories(
    runtime: PolicyRuntime,
    response: RankResponse,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in response.ranked_candidates[:10]:
        item = runtime.news.get(candidate.news_id)
        if item is None:
            continue
        category = item.category or "<missing>"
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


app = create_app()
