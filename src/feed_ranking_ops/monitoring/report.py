from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from feed_ranking_ops.evaluation.processed import load_processed_dataset
from feed_ranking_ops.monitoring.diagnostics import (
    build_input_profile,
    candidate_count_bucket,
    compare_profiles,
    compute_policy_slice_diagnostics,
    history_length_bucket,
    jensen_shannon_divergence,
    normalize_distribution,
)
from feed_ranking_ops.serving.policy import load_policy_runtime
from feed_ranking_ops.serving.schemas import RankRequestLogEvent


def generate_monitoring_report(
    *,
    processed_dir: Path,
    serving_artifacts_dir: Path,
    reports_dir: Path,
    request_log: Path | None = None,
    limit_impressions: int | None = None,
) -> dict[str, Any]:
    if limit_impressions is not None and limit_impressions <= 0:
        raise ValueError("limit_impressions must be positive when provided")
    runtime = load_policy_runtime(
        serving_artifacts_dir / "policy_manifest.json"
    )
    dataset = load_processed_dataset(
        processed_dir,
        limit_impressions=limit_impressions,
    )
    request_summary, request_profile = summarize_request_log(request_log)
    reference_profile = build_input_profile(
        dataset.behaviors["train"],
        runtime,
    )
    comparison_profiles = {
        partition: build_input_profile(dataset.behaviors[partition], runtime)
        for partition in ("validation", "test")
    }
    drift_checks = {
        partition: compare_profiles(reference_profile, profile)
        for partition, profile in comparison_profiles.items()
    }
    warnings = list(request_summary["warnings"])
    if request_profile is not None:
        drift_checks["request_log"] = compare_request_profile(
            reference_profile,
            request_profile,
        )
        warnings.append(
            "Request-log category counts describe top-10 ranked results and are "
            "not directly compared with full candidate-category distributions."
        )
    else:
        warnings.append(
            "No serving request log was available; request-distribution drift was not computed."
        )
    report = {
        "selected_policy": runtime.manifest.selected_policy_name,
        "policy_family": runtime.manifest.selected_policy_family,
        "data_protocol": runtime.manifest.data_protocol,
        "final_partition_type": runtime.manifest.final_partition_type,
        "limit_impressions_per_partition": limit_impressions,
        "request_log": request_summary,
        "offline_policy_diagnostics": {
            partition: compute_policy_slice_diagnostics(
                dataset.behaviors[partition],
                runtime,
            )
            for partition in ("validation", "test")
        },
        "reference_profile": reference_profile,
        "comparison_profiles": comparison_profiles,
        "drift_checks": drift_checks,
        "warnings": sorted(set(warnings)),
        "drift_interpretation": (
            "These are offline distribution diagnostics, not continuous production monitoring."
        ),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "monitoring_report.json"
    markdown_path = reports_dir / "monitoring_report.md"
    _write_json(json_path, report)
    markdown_path.write_text(
        render_monitoring_report(report),
        encoding="utf-8",
    )
    return {
        "report": report,
        "outputs": {
            "monitoring_report_json": str(json_path),
            "monitoring_report_markdown": str(markdown_path),
        },
    }


def summarize_request_log(
    request_log: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if request_log is None or not request_log.is_file():
        return (
            {
                "available": False,
                "request_count": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "warnings": [],
            },
            None,
        )
    events: list[RankRequestLogEvent] = []
    warnings: list[str] = []
    for line_number, line in enumerate(
        request_log.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            events.append(RankRequestLogEvent.model_validate_json(line))
        except ValidationError:
            warnings.append(
                f"Skipped malformed request log event at line {line_number}."
            )
    latencies = [event.latency_ms for event in events]
    candidate_sizes = [event.candidate_id_count for event in events]
    ranked_sizes = [event.ranked_candidate_count for event in events]
    missing_total = sum(event.missing_candidate_count for event in events)
    unknown_total = sum(event.unknown_history_count for event in events)
    history_total = sum(event.history_id_count for event in events)
    candidate_total = sum(candidate_sizes)
    empty_count = sum(event.empty_history for event in events)
    category_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    history_buckets: Counter[str] = Counter()
    candidate_buckets: Counter[str] = Counter()
    for event in events:
        category_counts.update(event.top_ranked_category_counts)
        outcome_counts[event.outcome] += 1
        warning_counts.update(event.warnings)
        history_buckets[history_length_bucket(event.history_id_count)] += 1
        candidate_buckets[candidate_count_bucket(event.candidate_id_count)] += 1
    summary = {
        "available": True,
        "request_count": len(events),
        "successful_requests": sum(
            event.status == "success" for event in events
        ),
        "failed_requests": sum(event.status == "failed" for event in events),
        "latency_ms": _numeric_summary(latencies),
        "missing_candidates": {
            "count": missing_total,
            "rate": _rate(missing_total, candidate_total),
        },
        "unknown_history": {
            "count": unknown_total,
            "rate": _rate(unknown_total, history_total),
        },
        "empty_history": {
            "count": empty_count,
            "rate": _rate(empty_count, len(events)),
        },
        "candidate_list_size": _numeric_summary(candidate_sizes),
        "ranked_list_size": _numeric_summary(ranked_sizes),
        "top_ranked_category_distribution": normalize_distribution(
            category_counts
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "warnings": warnings,
    }
    profile = {
        "history_length_distribution": normalize_distribution(history_buckets),
        "candidate_count_distribution": normalize_distribution(
            candidate_buckets
        ),
        "empty_history_rate": _rate(empty_count, len(events)),
        "missing_candidate_rate": _rate(missing_total, candidate_total),
    }
    return summary, profile


def compare_request_profile(
    reference: dict[str, Any],
    request_profile: dict[str, Any],
) -> dict[str, float]:
    return {
        "history_length_js_divergence": jensen_shannon_divergence(
            reference["history_length_distribution"],
            request_profile["history_length_distribution"],
        ),
        "candidate_count_js_divergence": jensen_shannon_divergence(
            reference["candidate_count_distribution"],
            request_profile["candidate_count_distribution"],
        ),
        "empty_history_rate_absolute_difference": abs(
            float(reference["empty_history_rate"])
            - float(request_profile["empty_history_rate"])
        ),
        "missing_candidate_rate_absolute_difference": abs(
            float(reference["missing_candidate_rate"])
            - float(request_profile["missing_candidate_rate"])
        ),
    }


def render_monitoring_report(report: dict[str, Any]) -> str:
    request_log = report["request_log"]
    lines = [
        "# Serving And Offline Policy Monitoring",
        "",
        "## Policy",
        "",
        f"- Selected policy: `{report['selected_policy']}`",
        f"- Policy family: `{report['policy_family']}`",
        f"- Data protocol: `{report['data_protocol']}`",
        f"- Impression limit per partition: "
        f"`{report['limit_impressions_per_partition']}`",
        "",
        "## Request Log",
        "",
        f"- Available: `{str(request_log['available']).lower()}`",
        f"- Requests: {request_log['request_count']}",
        f"- Successful: {request_log['successful_requests']}",
        f"- Failed: {request_log['failed_requests']}",
    ]
    if request_log["available"]:
        lines.extend(
            [
                f"- Average latency: {_format(request_log['latency_ms']['mean'])} ms",
                f"- P95 latency: {_format(request_log['latency_ms']['p95'])} ms",
                f"- Missing candidate rate: "
                f"{request_log['missing_candidates']['rate']:.4f}",
                f"- Unknown history rate: "
                f"{request_log['unknown_history']['rate']:.4f}",
                f"- Empty history rate: "
                f"{request_log['empty_history']['rate']:.4f}",
            ]
        )
    lines.extend(["", "## Offline Policy Metrics", ""])
    lines.extend(
        [
            "| Partition | Impressions | MRR | NDCG@5 | NDCG@10 | AUC | Tie fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for partition, diagnostics in report["offline_policy_diagnostics"].items():
        overall = diagnostics["overall"]
        metrics = overall["metrics"]
        lines.append(
            f"| {partition} | {overall['n_impressions']} | "
            f"{_format(metrics.get('mrr'))} | "
            f"{_format(metrics.get('ndcg@5'))} | "
            f"{_format(metrics.get('ndcg@10'))} | "
            f"{_format(metrics.get('auc'))} | "
            f"{diagnostics['score_ties']['fraction']:.4f} |"
        )
    lines.extend(["", "## Drift-Style Checks", ""])
    for partition, checks in report["drift_checks"].items():
        lines.append(f"### {partition}")
        lines.append("")
        for name, value in checks.items():
            lines.append(f"- `{name}`: {float(value):.6f}")
        lines.append("")
    lines.extend(
        [
            "## Slice Diagnostics",
            "",
            "The JSON report includes metrics by history-length bucket, candidate-count "
            "bucket, empty/non-empty history, and top-ranked candidate category.",
            "",
            "## Warnings And Limitations",
            "",
            *[f"- {warning}" for warning in report["warnings"]],
            f"- {report['drift_interpretation']}",
            "- Request logs contain aggregate counts and category summaries, not raw IDs.",
        ]
    )
    return "\n".join(lines) + "\n"


def _numeric_summary(values: list[float | int]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(numeric)),
        "mean": float(np.mean(numeric)),
        "p50": float(np.percentile(numeric, 50)),
        "p95": float(np.percentile(numeric, 95)),
        "max": float(np.max(numeric)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
