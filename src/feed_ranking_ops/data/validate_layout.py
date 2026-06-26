from __future__ import annotations

import argparse
from pathlib import Path

from feed_ranking_ops.data.layout import (
    DATA_PROTOCOLS,
    DEFAULT_DATA_PROTOCOL,
    format_layout_validation,
    validate_mind_layout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate expected MIND-small data layout.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source folders required by the selected protocol.",
    )
    parser.add_argument(
        "--protocol",
        choices=DATA_PROTOCOLS,
        default=DEFAULT_DATA_PROTOCOL,
        help="Source layout protocol to validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_mind_layout(args.data_dir, protocol=args.protocol)
    print(format_layout_validation(result))
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
