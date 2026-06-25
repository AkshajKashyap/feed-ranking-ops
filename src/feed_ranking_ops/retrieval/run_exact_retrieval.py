from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.evaluation.processed import ProcessedDataError
from feed_ranking_ops.retrieval.protocol import (
    DEFAULT_DECAYS,
    DEFAULT_HISTORY_LENGTHS,
    DEFAULT_TEXT_CONFIGS,
    run_exact_retrieval_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exact full-catalog retrieval evaluation.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/retrieval"))
    parser.add_argument(
        "--catalog-protocol",
        choices=["observed_available", "static_partition_catalog"],
        default="observed_available",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--text-configs", default=",".join(DEFAULT_TEXT_CONFIGS))
    parser.add_argument(
        "--history-lengths",
        default=",".join("all" if value is None else str(value) for value in DEFAULT_HISTORY_LENGTHS),
    )
    parser.add_argument("--decay-values", default=",".join(f"{value:g}" for value in DEFAULT_DECAYS))
    parser.add_argument("--exclude-history", default="true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_exact_retrieval_protocol(
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            catalog_protocol=args.catalog_protocol,
            top_k=args.top_k,
            limit_queries=args.limit_queries,
            text_configs=_parse_csv(args.text_configs),
            history_lengths=_parse_history_lengths(args.history_lengths),
            decay_values=_parse_floats(args.decay_values, "--decay-values"),
            exclude_history=_parse_bool(args.exclude_history),
            seed=args.seed,
        )
    except (FileNotFoundError, ProcessedDataError, ValueError) as exc:
        raise SystemExit(f"Exact retrieval evaluation failed: {exc}") from exc
    print("Completed exact full-catalog retrieval evaluation.")
    print(f"Smoke test: {result['protocol']['smoke_test']}")
    print(f"Selected configuration: {result['protocol']['selected_configuration_name']}")
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated option must contain at least one value")
    return values


def _parse_history_lengths(value: str) -> list[int | None]:
    lengths: list[int | None] = []
    for item in _parse_csv(value):
        if item.lower() == "all":
            lengths.append(None)
        else:
            parsed = int(item)
            if parsed <= 0:
                raise ValueError("--history-lengths values must be positive or all")
            lengths.append(parsed)
    return lengths


def _parse_floats(value: str, option_name: str) -> list[float]:
    try:
        values = [float(item) for item in _parse_csv(value)]
    except ValueError as exc:
        raise ValueError(f"{option_name} must contain comma-separated numbers") from exc
    if any(parsed <= 0 for parsed in values):
        raise ValueError(f"{option_name} values must be positive")
    return values


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError("--exclude-history must be true or false")


if __name__ == "__main__":
    raise SystemExit(main())
