import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI

from feed_ranking_ops.evaluation.baselines import CategoryAffinityBaseline
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)
from feed_ranking_ops.serving.app import create_app
from feed_ranking_ops.serving.schemas import RankRequest

from test_policy_selection import _select


def _app(tmp_path: Path) -> FastAPI:
    _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.352,
        threshold=0.01,
    )
    return create_app(tmp_path / "serving" / "policy_manifest.json")


def test_health_and_policy_endpoints(tmp_path: Path):
    health, policy = _requests(
        _app(tmp_path),
        [("GET", "/health", None), ("GET", "/policy", None)],
    )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "policy_loaded": True,
        "selected_policy": "category_affinity",
        "error": None,
    }
    assert policy.status_code == 200
    assert policy.json()["selected_policy_name"] == "category_affinity"
    assert policy.json()["fitting_partitions"] == ["train", "validation"]
    assert "artifact_paths" not in policy.json()


def test_rank_matches_offline_category_affinity_semantics(tmp_path: Path):
    (response,) = _requests(
        _app(tmp_path),
        [
            (
                "POST",
                "/rank",
                {
                    "history_news_ids": ["N1"],
                    "candidate_news_ids": ["N2", "N1", "N4"],
                },
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    score_by_id = {
        row["news_id"]: row["score"] for row in payload["ranked_candidates"]
    }
    news = {
        "N1": NewsItem("N1", "news", "local", "", ""),
        "N2": NewsItem("N2", "sports", "football", "", ""),
        "N4": NewsItem("N4", "tech", "ai", "", ""),
    }
    behavior = BehaviorImpression(
        partition="serving",
        impression_id="request",
        user_id="anonymous",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        history_news_ids=["N1"],
        candidates=[
            ImpressionCandidate(0, "N2", None),
            ImpressionCandidate(1, "N1", None),
            ImpressionCandidate(2, "N4", None),
        ],
    )
    expected = CategoryAffinityBaseline().score(behavior, news).scores

    assert score_by_id == dict(
        zip(["N2", "N1", "N4"], expected, strict=True)
    )
    assert [row["news_id"] for row in payload["ranked_candidates"]] == [
        "N1",
        "N2",
        "N4",
    ]


def test_rank_handles_unknown_candidates_and_history(tmp_path: Path):
    (response,) = _requests(
        _app(tmp_path),
        [
            (
                "POST",
                "/rank",
                {
                    "history_news_ids": ["N1", "UNKNOWN_HISTORY"],
                    "candidate_news_ids": ["UNKNOWN_CANDIDATE", "N2"],
                },
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_candidate_ids"] == ["UNKNOWN_CANDIDATE"]
    assert payload["unknown_history_ids"] == ["UNKNOWN_HISTORY"]
    assert [row["news_id"] for row in payload["ranked_candidates"]] == ["N2"]
    assert len(payload["warnings"]) >= 2


def test_empty_history_and_repeated_calls_are_deterministic(tmp_path: Path):
    request = {
        "history_news_ids": [],
        "candidate_news_ids": ["N2", "N1", "N4"],
    }
    first, second = _requests(
        _app(tmp_path),
        [("POST", "/rank", request), ("POST", "/rank", request)],
    )

    assert first.status_code == 200
    assert first.json() == second.json()
    assert [row["news_id"] for row in first.json()["ranked_candidates"]] == [
        "N2",
        "N1",
        "N4",
    ]
    assert "History is empty" in " ".join(first.json()["warnings"])


def test_missing_policy_is_degraded_and_returns_service_unavailable(tmp_path: Path):
    health, policy, ranking = _requests(
        create_app(tmp_path / "missing.json"),
        [
            ("GET", "/health", None),
            ("GET", "/policy", None),
            (
                "POST",
                "/rank",
                {"history_news_ids": [], "candidate_news_ids": ["N1"]},
            ),
        ],
    )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert policy.status_code == 503
    assert ranking.status_code == 503


def test_rank_request_rejects_empty_candidate_list():
    try:
        RankRequest(history_news_ids=[], candidate_news_ids=[])
    except ValueError as exc:
        assert "at least 1 item" in str(exc)
    else:
        raise AssertionError("empty candidate list should fail validation")


def test_manifest_catalog_schema_mismatch_fails_closed(tmp_path: Path):
    application = _app(tmp_path)
    manifest_path = tmp_path / "serving" / "policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["news_catalog_columns"] = ["news_id"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (health,) = _requests(application, [("GET", "/health", None)])

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["policy_loaded"] is False
    assert "schema" in health.json()["error"].lower()


def _requests(
    application: FastAPI,
    requests: list[tuple[str, str, dict | None]],
) -> list[httpx.Response]:
    return asyncio.run(_async_requests(application, requests))


async def _async_requests(
    application: FastAPI,
    requests: list[tuple[str, str, dict | None]],
) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=application)
    responses = []
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for method, path, payload in requests:
                responses.append(
                    await client.request(method, path, json=payload)
                )
    return responses
