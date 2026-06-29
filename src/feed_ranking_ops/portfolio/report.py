from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feed_ranking_ops import __version__


def generate_portfolio_report(
    *,
    reports_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    missing: list[str] = []
    inputs = {
        "baseline_validation": _optional_json(
            reports_dir / "baselines" / "validation_metrics.json", missing
        ),
        "baseline_test": _optional_json(
            reports_dir / "baselines" / "test_metrics.json", missing
        ),
        "exact_protocol": _optional_json(
            reports_dir / "retrieval_smoke_100" / "protocol.json", missing
        ),
        "exact_validation": _optional_json(
            reports_dir / "retrieval_smoke_100" / "validation_metrics.json",
            missing,
        ),
        "exact_test": _optional_json(
            reports_dir / "retrieval_smoke_100" / "test_metrics.json", missing
        ),
        "ann_protocol": _optional_json(
            reports_dir / "ann_1000_flat" / "protocol.json", missing
        ),
        "ann_validation": _optional_json(
            reports_dir / "ann_1000_flat" / "validation_metrics.json", missing
        ),
        "ann_test": _optional_json(
            reports_dir / "ann_1000_flat" / "test_metrics.json", missing
        ),
        "ltr_validation": _optional_json(
            reports_dir / "ltr" / "validation_metrics.json", missing
        ),
        "ltr_test": _optional_json(
            reports_dir / "ltr" / "test_metrics.json", missing
        ),
        "promotion": _optional_json(
            reports_dir / "model_selection" / "promotion_report.json", missing
        ),
        "monitoring": _optional_json(
            reports_dir / "monitoring" / "monitoring_report.json", missing
        ),
        "manifest": _optional_json(
            artifacts_dir / "serving" / "policy_manifest.json", missing
        ),
    }
    summary = _build_summary(inputs, missing)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_summary.json"
    markdown_path = output_dir / "portfolio_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_portfolio_summary(summary),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "outputs": {
            "portfolio_summary_json": str(json_path),
            "portfolio_summary_markdown": str(markdown_path),
        },
    }


def render_portfolio_summary(summary: dict[str, Any]) -> str:
    baseline = summary["logged_candidate_baseline"]
    ltr = summary["learning_to_rank"]
    exact = summary["exact_retrieval_smoke"]
    ann = summary["ann_benchmark"]
    monitoring = summary["monitoring"]
    lines = [
        "# FeedRank Ops Portfolio Summary",
        "",
        "A leakage-aware Microsoft MIND-small recommendation project covering "
        "chronological preparation, retrieval, ranking, promotion, serving, and "
        "offline monitoring.",
        "",
        "> Internal-test values use a train-only chronological holdout and are not "
        "official MIND validation results.",
        "",
        "## Results At A Glance",
        "",
        "| Component | Result |",
        "| --- | --- |",
        f"| Best logged-candidate baseline | {_metric_result(baseline)} |",
        f"| Pointwise LTR | {_metric_result(ltr)} |",
        f"| Exact retrieval smoke | {_runtime_result(exact)} |",
        f"| Batched FAISS ANN | {_runtime_result(ann)} |",
        f"| Selected serving policy | "
        f"{summary['promotion'].get('selected_policy') or 'Unavailable'} |",
        f"| Monitoring | {_monitoring_result(monitoring)} |",
        "",
        "## Data And Protocol",
        "",
        f"- Dataset: {summary['project']['dataset']}",
        f"- Protocol: {summary['data'].get('protocol') or 'Unavailable'}",
        f"- Split rows: {summary['data'].get('split_counts') or 'Unavailable'}",
        "",
        "## Promotion Decision",
        "",
        f"- Decision: {summary['promotion'].get('decision') or 'Unavailable'}",
        f"- Learned ranker: "
        f"{summary['promotion'].get('learned_result') or 'Unavailable'}",
        f"- Observed validation lift: "
        f"{_percent(summary['promotion'].get('observed_relative_improvement'))}",
        f"- Required lift: "
        f"{_percent(summary['promotion'].get('threshold_relative'))}",
        "",
        "## Serving",
        "",
        f"- Endpoints: {', '.join(summary['serving']['endpoints'])}",
        f"- Request logging: {summary['serving']['request_logging']}",
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in summary["limitations"]],
        "",
        "## Recommended Next Work",
        "",
        *[f"- {value}" for value in summary["recommended_next_work"]],
    ]
    if summary["availability"]["missing_inputs"]:
        lines.extend(
            [
                "",
                "## Unavailable Inputs",
                "",
                *[
                    f"- {path}"
                    for path in summary["availability"]["missing_inputs"]
                ],
            ]
        )
    return "\n".join(lines) + "\n"


def _build_summary(
    inputs: dict[str, dict[str, Any] | None],
    missing: list[str],
) -> dict[str, Any]:
    baseline = _ranker_summary(
        inputs["baseline_validation"],
        inputs["baseline_test"],
        group="baselines",
    )
    ltr = _ranker_summary(
        inputs["ltr_validation"],
        inputs["ltr_test"],
        group="rankers",
        selected_key="selected_model_name",
    )
    manifest = inputs["manifest"]
    promotion = inputs["promotion"]
    monitoring = inputs["monitoring"]
    return {
        "project": {
            "name": "feed-ranking-ops",
            "version": __version__,
            "dataset": "Microsoft MIND-small",
        },
        "availability": {
            "complete": not missing,
            "missing_inputs": sorted(missing),
        },
        "data": {
            "available": manifest is not None,
            "protocol": _get(manifest, "data_protocol"),
            "final_partition_type": _get(manifest, "final_partition_type"),
            "split_counts": _get(inputs["baseline_test"], "partition_sizes"),
        },
        "logged_candidate_baseline": baseline,
        "exact_retrieval_smoke": _retrieval_summary(
            inputs["exact_protocol"],
            inputs["exact_validation"],
            inputs["exact_test"],
        ),
        "ann_benchmark": _ann_summary(
            inputs["ann_protocol"],
            inputs["ann_validation"],
            inputs["ann_test"],
        ),
        "learning_to_rank": ltr,
        "promotion": {
            "available": promotion is not None or manifest is not None,
            "selected_policy": _get(promotion, "selected_policy_name")
            or _get(manifest, "selected_policy_name"),
            "decision": _get(promotion, "promotion_decision")
            or _get(manifest, "promotion_decision"),
            "learned_result": _get(promotion, "learned_promotion_result")
            or _get(manifest, "learned_promotion_result"),
            "observed_relative_improvement": _get(
                promotion, "observed_validation_improvement_relative"
            )
            or _get(manifest, "observed_validation_improvement_relative"),
            "threshold_relative": _nested(
                promotion,
                "promotion_rule",
                "minimum_relative_improvement",
            )
            or _get(manifest, "promotion_threshold_relative"),
        },
        "serving": {
            "available": manifest is not None,
            "endpoints": ["/health", "/policy", "/rank", "/metrics"],
            "selected_policy": _get(manifest, "selected_policy_name"),
            "request_logging": (
                "Optional aggregate JSONL; raw history and candidate IDs are not logged."
            ),
        },
        "monitoring": _monitoring_summary(monitoring),
        "limitations": [
            "Internal-test metrics use a train-only chronological holdout, not official MIND validation.",
            "Logged-candidate metrics inherit exposure and position bias.",
            "Exact retrieval is a bounded correctness smoke, not a scalable production path.",
            "ANN-only runs without dense exact comparison do not report approximation recall.",
            "Category affinity produces frequent score ties and uses source position as tie-breaker.",
            "Serving telemetry and drift checks are local/offline diagnostics.",
        ],
        "recommended_next_work": [
            "Evaluate the protocol on an additional public temporal recommendation dataset.",
            "Package a future learned ranker only with complete fitted preprocessing state.",
            "Add hosted deployment only when a real operating environment exists.",
        ],
    }


def _ranker_summary(
    validation: dict[str, Any] | None,
    test: dict[str, Any] | None,
    *,
    group: str,
    selected_key: str | None = None,
) -> dict[str, Any]:
    candidates = _get(validation, group)
    if not isinstance(candidates, dict) or not candidates:
        return {"available": False}
    selected = _get(validation, selected_key) if selected_key else None
    if not isinstance(selected, str):
        selected = sorted(
            candidates,
            key=lambda name: (
                -_metric_value(candidates[name], "ndcg@10"),
                -_metric_value(candidates[name], "mrr"),
                name,
            ),
        )[0]
    test_candidates = _get(test, group)
    return {
        "available": True,
        "name": selected,
        "validation_metrics": _nested(candidates.get(selected), "metrics"),
        "internal_test_metrics": (
            _nested(test_candidates.get(selected), "metrics")
            if isinstance(test_candidates, dict)
            else None
        ),
    }


def _retrieval_summary(
    protocol: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    test: dict[str, Any] | None,
) -> dict[str, Any]:
    if protocol is None:
        return {"available": False}
    return {
        "available": True,
        "purpose": "bounded sparse exact correctness reference",
        "query_count": _nested(protocol, "timing", "number_of_queries"),
        "total_runtime_seconds": _nested(
            protocol, "timing", "total_runtime_seconds"
        ),
        "validation_metrics": _get(validation, "metrics"),
        "internal_test_metrics": _get(test, "metrics"),
    }


def _ann_summary(
    protocol: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    test: dict[str, Any] | None,
) -> dict[str, Any]:
    if protocol is None:
        return {"available": False}
    return {
        "available": True,
        "backend": _get(protocol, "backend"),
        "mode": _get(protocol, "mode"),
        "query_count": _nested(protocol, "timing", "number_of_queries"),
        "total_runtime_seconds": _nested(
            protocol, "timing", "total_runtime_seconds"
        ),
        "faiss_search_seconds": _nested(
            protocol, "timing", "faiss_search_seconds"
        ),
        "dense_exact_comparison_skipped": _get(
            protocol, "dense_exact_comparison_skipped"
        ),
        "validation_metrics": _get(validation, "metrics"),
        "internal_test_metrics": _get(test, "metrics"),
    }


def _monitoring_summary(
    monitoring: dict[str, Any] | None,
) -> dict[str, Any]:
    if monitoring is None:
        return {"available": False}
    return {
        "available": True,
        "request_count": _nested(monitoring, "request_log", "request_count"),
        "validation_ndcg_at_10": _nested(
            monitoring,
            "offline_policy_diagnostics",
            "validation",
            "overall",
            "metrics",
            "ndcg@10",
        ),
        "internal_test_ndcg_at_10": _nested(
            monitoring,
            "offline_policy_diagnostics",
            "test",
            "overall",
            "metrics",
            "ndcg@10",
        ),
        "validation_tie_fraction": _nested(
            monitoring,
            "offline_policy_diagnostics",
            "validation",
            "score_ties",
            "fraction",
        ),
        "internal_test_tie_fraction": _nested(
            monitoring,
            "offline_policy_diagnostics",
            "test",
            "score_ties",
            "fraction",
        ),
    }


def _optional_json(
    path: Path,
    missing: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(str(path))
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        missing.append(str(path))
        return None
    if not isinstance(value, dict):
        missing.append(str(path))
        return None
    return value


def _metric_value(summary: Any, name: str) -> float:
    value = _nested(summary, "metrics", name)
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _get(value: Any, key: str | None) -> Any:
    return value.get(key) if isinstance(value, dict) and key else None


def _metric_result(summary: dict[str, Any]) -> str:
    if not summary.get("available"):
        return "Unavailable"
    validation = _get(summary.get("validation_metrics"), "ndcg@10")
    test = _get(summary.get("internal_test_metrics"), "ndcg@10")
    return (
        f"`{summary.get('name', 'policy')}`; validation NDCG@10 "
        f"{_number(validation)}, internal-test {_number(test)}"
    )


def _runtime_result(summary: dict[str, Any]) -> str:
    if not summary.get("available"):
        return "Unavailable"
    return (
        f"{summary.get('query_count', 'N/A')} queries in "
        f"{_number(summary.get('total_runtime_seconds'))} s"
    )


def _monitoring_result(summary: dict[str, Any]) -> str:
    if not summary.get("available"):
        return "Unavailable"
    return (
        f"{summary.get('request_count', 0)} logged smoke requests; "
        f"validation tie fraction "
        f"{_percent(summary.get('validation_tie_fraction'))}"
    )


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"
