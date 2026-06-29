from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.ranking.selection import (
    DEFAULT_PROMOTION_THRESHOLD,
    select_and_package_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and package a logged-candidate serving policy."
    )
    parser.add_argument(
        "--baseline-reports-dir",
        type=Path,
        default=Path("reports/baselines"),
    )
    parser.add_argument(
        "--ltr-reports-dir",
        type=Path,
        default=Path("reports/ltr"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/model_selection"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/serving"),
    )
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=DEFAULT_PROMOTION_THRESHOLD,
        help="Required validation NDCG@10 lift over the strongest baseline.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = select_and_package_policy(
            baseline_reports_dir=args.baseline_reports_dir,
            ltr_reports_dir=args.ltr_reports_dir,
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            artifacts_dir=args.artifacts_dir,
            minimum_relative_improvement=args.minimum_relative_improvement,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Policy selection failed: {exc}") from exc
    report = result["report"]
    print(f"Selected policy: {report['selected_policy_name']}")
    print(f"Promotion decision: {report['promotion_decision']}")
    print(f"Learned ranker: {report['learned_promotion_result']}")
    print(report["rationale"])
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
