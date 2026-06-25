from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.evaluation.baselines import default_baseline_names
from feed_ranking_ops.evaluation.protocol import DEFAULT_HALF_LIVES, run_baseline_protocol
from feed_ranking_ops.evaluation.processed import ProcessedDataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate non-neural logged-candidate ranking baselines."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/baselines"))
    parser.add_argument(
        "--limit-impressions",
        type=int,
        default=None,
        help="Optional smoke-test limit per partition. Results are clearly labeled.",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(default_baseline_names()),
        help="Comma-separated baselines to run.",
    )
    parser.add_argument(
        "--half-lives",
        default=",".join(f"{value:g}" for value in DEFAULT_HALF_LIVES),
        help="Comma-separated time-decay half-lives in hours.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baselines = _parse_names(args.baselines)
        half_lives = _parse_half_lives(args.half_lives)
        result = run_baseline_protocol(
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            baseline_names=baselines,
            half_life_hours=half_lives,
            limit_impressions=args.limit_impressions,
            seed=args.seed,
        )
    except (FileNotFoundError, ProcessedDataError, ValueError) as exc:
        raise SystemExit(f"Baseline evaluation failed: {exc}") from exc

    protocol = result["protocol"]
    print("Completed logged-candidate baseline evaluation.")
    print(f"Smoke test: {protocol['smoke_test']}")
    print(f"Selected hyperparameters: {protocol['selected_hyperparameters']}")
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


def _parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--baselines must include at least one baseline name")
    return names


def _parse_half_lives(value: str) -> list[float]:
    try:
        half_lives = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--half-lives must contain comma-separated numbers") from exc
    if not half_lives or any(value <= 0 for value in half_lives):
        raise ValueError("--half-lives must contain positive values")
    return half_lives


if __name__ == "__main__":
    raise SystemExit(main())
