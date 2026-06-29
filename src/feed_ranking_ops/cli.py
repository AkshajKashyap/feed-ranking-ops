from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from feed_ranking_ops import __version__
from feed_ranking_ops.serving.schemas import PolicyManifest

DEFAULT_MANIFEST = Path("artifacts/serving/policy_manifest.json")
MAJOR_CLIS = [
    "data.prepare_dataset",
    "evaluation.run_baselines",
    "retrieval.run_exact_retrieval",
    "retrieval.run_ann_benchmark",
    "ranking.run_ltr",
    "ranking.select_policy",
    "monitoring.generate_report",
    "portfolio.generate_report",
    "serving.app",
]
KEY_DOCS = [
    "README.md",
    "docs/architecture.md",
    "docs/model_card.md",
    "docs/experimental_methodology.md",
    "docs/release_checklist.md",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feed-ranking-ops",
        description="FeedRank Ops project utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    info = subparsers.add_parser(
        "project-info",
        help="Show release, policy, CLI, documentation, and artifact information.",
    )
    info.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    info.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable project information.",
    )
    return parser


def project_info(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_status = "unavailable"
    selected_policy = None
    data_protocol = None
    if manifest_path.is_file():
        try:
            manifest = PolicyManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest_status = "available"
            selected_policy = manifest.selected_policy_name
            data_protocol = manifest.data_protocol
        except (OSError, ValueError):
            manifest_status = "invalid"
    return {
        "package": "feed-ranking-ops",
        "version": __version__,
        "supported_task": "personalized news candidate retrieval and logged-candidate ranking",
        "dataset": "Microsoft MIND-small",
        "selected_policy": selected_policy,
        "data_protocol": data_protocol,
        "manifest_status": manifest_status,
        "major_clis": list(MAJOR_CLIS),
        "key_docs": list(KEY_DOCS),
        "artifact_locations": {
            "serving": "artifacts/serving/",
            "request_logs": "artifacts/logs/",
            "reports": "reports/",
            "portfolio": "reports/portfolio/",
        },
    }


def render_project_info(info: dict[str, Any]) -> str:
    selected_policy = info["selected_policy"] or "unavailable"
    protocol = info["data_protocol"] or "unavailable"
    lines = [
        f"{info['package']} {info['version']}",
        f"Task: {info['supported_task']}",
        f"Dataset: {info['dataset']}",
        f"Selected policy: {selected_policy}",
        f"Data protocol: {protocol}",
        f"Serving manifest: {info['manifest_status']}",
        "Major CLIs:",
        *[f"  - python -m feed_ranking_ops.{name}" for name in info["major_clis"]],
        "Key documentation:",
        *[f"  - {path}" for path in info["key_docs"]],
        "Artifact locations:",
        *[
            f"  - {name}: {path}"
            for name, path in info["artifact_locations"].items()
        ],
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "project-info":
        info = project_info(args.manifest)
        if args.json:
            print(json.dumps(info, indent=2, sort_keys=True))
        else:
            print(render_project_info(info))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
