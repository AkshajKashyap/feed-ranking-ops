from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.evaluation.processed import ProcessedDataError
from feed_ranking_ops.ranking.protocol import run_ltr_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate pointwise logged-candidate rankers."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/ltr"))
    parser.add_argument(
        "--limit-impressions",
        type=int,
        default=None,
        help="Optional deterministic smoke-test limit per partition.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_ltr_protocol(
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            limit_impressions=args.limit_impressions,
            seed=args.seed,
        )
    except (FileNotFoundError, ProcessedDataError, ValueError) as exc:
        raise SystemExit(f"Learning-to-rank evaluation failed: {exc}") from exc

    protocol = result["protocol"]
    print("Completed pointwise logged-candidate learning-to-rank evaluation.")
    print(f"Smoke test: {protocol['smoke_test']}")
    print(f"Selected model: {protocol['selected_model']['model_name']}")
    print(f"Total runtime: {protocol['timing']['total_runtime_seconds']:.2f}s")
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
