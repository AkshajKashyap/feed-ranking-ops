from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.portfolio.report import generate_portfolio_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic portfolio summaries from existing reports."
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/portfolio"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_portfolio_report(
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
    )
    summary = result["summary"]
    print(f"Portfolio inputs complete: {summary['availability']['complete']}")
    if summary["availability"]["missing_inputs"]:
        print(
            f"Unavailable inputs: "
            f"{len(summary['availability']['missing_inputs'])}"
        )
    for name, path in result["outputs"].items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
