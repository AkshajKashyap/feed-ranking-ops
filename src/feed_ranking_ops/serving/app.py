from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from feed_ranking_ops.serving.policy import (
    PolicyLoadError,
    PolicyRuntime,
    load_policy_runtime,
    public_policy_metadata,
)
from feed_ranking_ops.serving.schemas import (
    HealthResponse,
    PolicyResponse,
    RankRequest,
    RankResponse,
)

DEFAULT_MANIFEST_PATH = Path("artifacts/serving/policy_manifest.json")


def create_app(manifest_path: Path | None = None) -> FastAPI:
    configured_path = manifest_path

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
        runtime = _require_runtime(request)
        return runtime.rank(payload)

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


app = create_app()
