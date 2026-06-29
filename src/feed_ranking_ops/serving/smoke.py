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

    health, policy, ranking = asyncio.run(
        _call_endpoints(args.manifest, news_ids)
    )
    for name, response in (
        ("health", health),
        ("policy", policy),
        ("rank", ranking),
    ):
        if response.status_code != 200:
            raise SystemExit(
                f"Serving smoke failed: {name} returned {response.status_code}: "
                f"{response.text}"
            )
    print(f"Health: {health.json()}")
    print(f"Policy: {policy.json()['selected_policy_name']}")
    print(f"Ranked candidates: {len(ranking.json()['ranked_candidates'])}")
    return 0


async def _call_endpoints(
    manifest_path: Path,
    news_ids: list[str],
) -> tuple[httpx.Response, httpx.Response, httpx.Response]:
    application = create_app(manifest_path)
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://serving-smoke",
        ) as client:
            health = await client.get("/health")
            policy = await client.get("/policy")
            ranking = await client.post(
                "/rank",
                json={
                    "history_news_ids": news_ids[:1],
                    "candidate_news_ids": news_ids,
                },
            )
    return health, policy, ranking


if __name__ == "__main__":
    raise SystemExit(main())
