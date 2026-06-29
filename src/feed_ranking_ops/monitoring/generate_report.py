from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.data.schemas import MindDataError
from feed_ranking_ops.monitoring.report import generate_monitoring_report
from feed_ranking_ops.serving.policy import PolicyLoadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate serving and offline policy monitoring diagnostics."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--serving-artifacts-dir",
        type=Path,
        default=Path("artifacts/serving"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/monitoring"),
    )
    parser.add_argument(
        "--request-log",
        type=Path,
        default=None,
        help="Optional JSONL rank request log.",
    )
    parser.add_argument(
        "--limit-impressions",
        type=int,
        default=None,
        help="Optional deterministic limit per partition for smoke reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_monitoring_report(
            processed_dir=args.processed_dir,
            serving_artifacts_dir=args.serving_artifacts_dir,
            reports_dir=args.reports_dir,
            request_log=args.request_log,
            limit_impressions=args.limit_impressions,
        )
    except (FileNotFoundError, MindDataError, PolicyLoadError, ValueError) as exc:
        raise SystemExit(f"Monitoring report generation failed: {exc}") from exc
    report = result["report"]
    print(f"Selected policy: {report['selected_policy']}")
    print(
        f"Request log available: {report['request_log']['available']} "
        f"({report['request_log']['request_count']} events)"
    )
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
