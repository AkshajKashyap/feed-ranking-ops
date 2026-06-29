from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from feed_ranking_ops.serving.app import create_app
from feed_ranking_ops.serving.schemas import PolicyManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the packaged ranking API.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/serving/policy_manifest.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"Serving smoke failed: missing manifest {args.manifest}")
    manifest = PolicyManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    catalog_name = manifest.artifact_paths.get("news_catalog")
    if not catalog_name:
        raise SystemExit("Serving smoke failed: manifest has no news catalog")
    catalog_path = args.manifest.parent / catalog_name
    if not catalog_path.is_file():
        raise SystemExit("Serving smoke failed: news catalog is missing")
    news_ids = (
        pq.read_table(catalog_path, columns=["news_id"])
        .column("news_id")
        .slice(0, 3)
        .to_pylist()
    )
    if not news_ids:
        raise SystemExit("Serving smoke failed: news catalog is empty")

    responses = asyncio.run(
        _call_endpoints(args.manifest, news_ids)
    )
    for name, response in responses.items():
        if response.status_code != 200:
            raise SystemExit(
                f"Serving smoke failed: {name} returned {response.status_code}: "
                f"{response.text}"
            )
    print(f"Health: {responses['health'].json()}")
    print(f"Policy: {responses['policy'].json()['selected_policy_name']}")
    print(
        f"Normal ranked candidates: "
        f"{len(responses['normal_rank'].json()['ranked_candidates'])}"
    )
    print(
        f"Empty-history warnings: "
        f"{len(responses['empty_history_rank'].json()['warnings'])}"
    )
    print(
        f"Unknown candidates: "
        f"{responses['unknown_candidate_rank'].json()['missing_candidate_ids']}"
    )
    print(f"Metrics: {responses['metrics'].json()}")
    return 0


async def _call_endpoints(
    manifest_path: Path,
    news_ids: list[str],
) -> dict[str, httpx.Response]:
    application = create_app(manifest_path)
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://serving-smoke",
        ) as client:
            responses = {
                "health": await client.get("/health"),
                "policy": await client.get("/policy"),
                "normal_rank": await client.post(
                    "/rank",
                    json={
                        "history_news_ids": news_ids[:1],
                        "candidate_news_ids": news_ids,
                    },
                ),
                "empty_history_rank": await client.post(
                    "/rank",
                    json={
                        "history_news_ids": [],
                        "candidate_news_ids": news_ids,
                    },
                ),
                "unknown_candidate_rank": await client.post(
                    "/rank",
                    json={
                        "history_news_ids": news_ids[:1],
                        "candidate_news_ids": [
                            "UNKNOWN_SMOKE_CANDIDATE",
                            *news_ids[:1],
                        ],
                    },
                ),
                "metrics": await client.get("/metrics"),
            }
    return responses


if __name__ == "__main__":
    raise SystemExit(main())
