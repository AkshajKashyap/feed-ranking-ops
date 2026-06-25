from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.evaluation.processed import ProcessedDataError
from feed_ranking_ops.retrieval.ann_protocol import (
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_HNSW_M,
    DEFAULT_OVERSAMPLING,
    DEFAULT_SVD_DIMS,
    run_ann_benchmark,
)
from feed_ranking_ops.retrieval.faiss_backend import FaissUnavailableError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dense exact and FAISS approximate full-catalog retrieval."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/ann"))
    parser.add_argument("--svd-dims", default=",".join(str(value) for value in DEFAULT_SVD_DIMS))
    parser.add_argument("--hnsw-m", default=",".join(str(value) for value in DEFAULT_HNSW_M))
    parser.add_argument(
        "--ef-construction",
        default=",".join(str(value) for value in DEFAULT_EF_CONSTRUCTION),
    )
    parser.add_argument("--ef-search", default=",".join(str(value) for value in DEFAULT_EF_SEARCH))
    parser.add_argument(
        "--oversampling",
        default=",".join(str(value) for value in DEFAULT_OVERSAMPLING),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument(
        "--catalog-protocol",
        choices=["observed_available", "static_partition_catalog"],
        default="observed_available",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--faiss-threads", type=int, default=None)
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--load-index", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_ann_benchmark(
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            svd_dims=_parse_ints(args.svd_dims, "--svd-dims"),
            hnsw_m_values=_parse_ints(args.hnsw_m, "--hnsw-m"),
            ef_construction_values=_parse_ints(args.ef_construction, "--ef-construction"),
            ef_search_values=_parse_ints(args.ef_search, "--ef-search"),
            oversampling_factors=_parse_ints(args.oversampling, "--oversampling"),
            top_k=args.top_k,
            limit_queries=args.limit_queries,
            catalog_protocol=args.catalog_protocol,
            seed=args.seed,
            faiss_threads=args.faiss_threads,
            save_index=args.save_index,
            load_index=args.load_index,
        )
    except FaissUnavailableError as exc:
        raise SystemExit(f"ANN benchmark failed: {exc}") from exc
    except (FileNotFoundError, ProcessedDataError, ValueError) as exc:
        raise SystemExit(f"ANN benchmark failed: {exc}") from exc
    print("Completed dense/FAISS ANN retrieval benchmark.")
    print(f"Smoke test: {result['protocol']['smoke_test']}")
    print(
        "Selected configuration: "
        f"{result['selected_configuration']['configuration_name']}"
    )
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


def _parse_ints(value: str, option_name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must contain comma-separated integers") from exc
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    if any(parsed <= 0 for parsed in values):
        raise ValueError(f"{option_name} values must be positive")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
