from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.processed import PROCESSED_FILES, load_news
from feed_ranking_ops.serving.schemas import PolicyManifest

DEFAULT_PROMOTION_THRESHOLD = 0.03
PRIMARY_METRIC = "ndcg@10"
TIEBREAKER_METRICS = ["mrr", "auc"]
SERVABLE_BASELINES = {"category_affinity", "original_order"}


def select_and_package_policy(
    *,
    baseline_reports_dir: Path,
    ltr_reports_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    artifacts_dir: Path,
    minimum_relative_improvement: float = DEFAULT_PROMOTION_THRESHOLD,
    created_at: str | None = None,
) -> dict[str, Any]:
    if minimum_relative_improvement < 0:
        raise ValueError("minimum_relative_improvement must be non-negative")
    baseline_validation = _read_json(
        baseline_reports_dir / "validation_metrics.json"
    )
    baseline_test = _read_json(baseline_reports_dir / "test_metrics.json")
    ltr_validation = _read_json(ltr_reports_dir / "validation_metrics.json")
    ltr_test = _read_json(ltr_reports_dir / "test_metrics.json")
    split_metadata = _read_json(processed_dir / PROCESSED_FILES["metadata"])

    baseline_name, baseline_validation_summary = _best_candidate(
        _required_mapping(baseline_validation, "baselines"),
    )
    baseline_test_summary = _matching_summary(
        baseline_name,
        baseline_validation_summary,
        _required_mapping(baseline_test, "baselines"),
    )
    learned_name = _selected_learned_name(ltr_validation)
    learned_validation_summary = _required_mapping(
        ltr_validation,
        "rankers",
    ).get(learned_name)
    if not isinstance(learned_validation_summary, dict):
        raise ValueError(
            f"LTR validation metrics do not contain selected model {learned_name!r}"
        )
    learned_test_summary = _matching_summary(
        learned_name,
        learned_validation_summary,
        _required_mapping(ltr_test, "rankers"),
    )

    baseline_validation_metrics = _metrics(baseline_validation_summary)
    baseline_test_metrics = _metrics(baseline_test_summary)
    learned_validation_metrics = _metrics(learned_validation_summary)
    learned_test_metrics = _metrics(learned_test_summary)
    baseline_primary = _required_metric(
        baseline_validation_metrics,
        PRIMARY_METRIC,
        baseline_name,
    )
    learned_primary = _required_metric(
        learned_validation_metrics,
        PRIMARY_METRIC,
        learned_name,
    )
    denominator = abs(baseline_primary)
    relative_improvement = (
        (learned_primary - baseline_primary) / denominator
        if denominator > 0
        else float("inf")
    )

    if relative_improvement >= minimum_relative_improvement:
        selected_name = learned_name
        selected_summary = learned_validation_summary
        selected_validation_metrics = learned_validation_metrics
        selected_test_metrics = learned_test_metrics
        selected_family = str(
            learned_validation_summary.get("model_family", "learned_ranker")
        )
        decision = "promote_learned_ranker"
        learned_result = "promoted"
        serving_ready = False
        rationale = (
            f"The learned ranker improved validation {PRIMARY_METRIC} by "
            f"{relative_improvement:.2%}, clearing the "
            f"{minimum_relative_improvement:.2%} promotion threshold."
        )
    else:
        selected_name = baseline_name
        selected_summary = baseline_validation_summary
        selected_validation_metrics = baseline_validation_metrics
        selected_test_metrics = baseline_test_metrics
        selected_family = str(
            baseline_validation_summary.get("baseline_family", baseline_name)
        )
        decision = "promote_baseline_policy"
        learned_result = "rejected_insufficient_improvement"
        serving_ready = selected_family in SERVABLE_BASELINES
        rationale = (
            f"The learned ranker improved validation {PRIMARY_METRIC} by "
            f"{relative_improvement:.2%}, below the "
            f"{minimum_relative_improvement:.2%} threshold. The simpler strongest "
            "validation baseline remains the serving policy."
        )

    internal_type = str(
        split_metadata.get("final_partition_type", "unknown")
    )
    holdout_warning = (
        str(split_metadata.get("comparability_warning"))
        if internal_type == "internal_chronological_holdout"
        else None
    )
    limitations = [
        "This service ranks only caller-supplied candidate sets; it does not retrieve from the full catalog.",
        "Offline click labels reflect exposure and position bias.",
        "Internal-test metrics are reported after selection and are not promotion inputs.",
    ]
    if decision == "promote_learned_ranker":
        limitations.append(
            "Learned-ranker serving is deferred because Milestone 5 did not package "
            "a fitted model and preprocessing state."
        )
    if holdout_warning:
        limitations.append(holdout_warning)

    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = artifacts_dir / "news_catalog.parquet"
    _write_news_catalog(processed_dir, catalog_path)
    fitting_partitions = _fitting_partitions(
        selected_family,
        selected_summary,
        baseline_test_summary if selected_name == baseline_name else learned_test_summary,
    )
    manifest = PolicyManifest(
        news_catalog_columns=["news_id", "category", "subcategory"],
        selected_policy_name=selected_name,
        selected_policy_family=selected_family,
        selected_metric=PRIMARY_METRIC,
        validation_metrics=selected_validation_metrics,
        internal_test_metrics=selected_test_metrics,
        promotion_decision=decision,
        learned_promotion_result=learned_result,
        promotion_threshold_relative=minimum_relative_improvement,
        observed_validation_improvement_relative=relative_improvement,
        created_at=created_at or _utc_timestamp(),
        git_commit=_git_commit(),
        data_protocol=str(split_metadata.get("protocol", "unknown")),
        final_partition_type=internal_type,
        internal_holdout_warning=holdout_warning,
        fitting_partitions=fitting_partitions,
        policy_config=_policy_config(selected_summary),
        artifact_paths={"news_catalog": catalog_path.name},
        serving_ready=serving_ready,
        limitations=limitations,
    )
    report = {
        "promotion_rule": {
            "primary_metric": PRIMARY_METRIC,
            "tie_breakers": TIEBREAKER_METRICS,
            "minimum_relative_improvement": minimum_relative_improvement,
            "internal_test_used_for_selection": False,
        },
        "strongest_baseline": {
            "policy_name": baseline_name,
            "policy_family": baseline_validation_summary.get(
                "baseline_family",
                baseline_name,
            ),
            "validation_metrics": baseline_validation_metrics,
            "internal_test_metrics": baseline_test_metrics,
        },
        "learned_candidate": {
            "policy_name": learned_name,
            "policy_family": learned_validation_summary.get(
                "model_family",
                "learned_ranker",
            ),
            "validation_metrics": learned_validation_metrics,
            "internal_test_metrics": learned_test_metrics,
        },
        "observed_validation_improvement_relative": relative_improvement,
        "promotion_decision": decision,
        "learned_promotion_result": learned_result,
        "selected_policy_name": selected_name,
        "selected_policy_family": selected_family,
        "rationale": rationale,
        "internal_holdout_warning": holdout_warning,
        "serving_ready": serving_ready,
    }
    report_json = reports_dir / "promotion_report.json"
    report_markdown = reports_dir / "promotion_report.md"
    manifest_path = artifacts_dir / "policy_manifest.json"
    _write_json(report_json, report)
    report_markdown.write_text(
        render_promotion_report(report),
        encoding="utf-8",
    )
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    return {
        "report": report,
        "manifest": manifest.model_dump(mode="json"),
        "outputs": {
            "promotion_report_json": str(report_json),
            "promotion_report_markdown": str(report_markdown),
            "policy_manifest": str(manifest_path),
            "news_catalog": str(catalog_path),
        },
    }


def render_promotion_report(report: dict[str, Any]) -> str:
    baseline = report["strongest_baseline"]
    learned = report["learned_candidate"]
    rule = report["promotion_rule"]
    lines = [
        "# Logged-Candidate Policy Promotion Report",
        "",
        "## Decision",
        "",
        f"- Selected policy: `{report['selected_policy_name']}`",
        f"- Policy family: `{report['selected_policy_family']}`",
        f"- Decision: `{report['promotion_decision']}`",
        f"- Learned-ranker result: `{report['learned_promotion_result']}`",
        f"- Rationale: {report['rationale']}",
        "",
        "## Promotion Rule",
        "",
        f"- Primary metric: validation `{rule['primary_metric']}`",
        f"- Tie-breakers: {', '.join(rule['tie_breakers'])}",
        f"- Minimum learned relative improvement: "
        f"{float(rule['minimum_relative_improvement']):.2%}",
        "- Internal-test metrics are reported after selection and never choose a policy.",
        "",
        "## Metric Comparison",
        "",
        "| Candidate | Split | NDCG@10 | MRR | AUC |",
        "| --- | --- | ---: | ---: | ---: |",
        _metric_row(
            str(baseline["policy_name"]),
            "validation",
            baseline["validation_metrics"],
        ),
        _metric_row(
            str(learned["policy_name"]),
            "validation",
            learned["validation_metrics"],
        ),
        _metric_row(
            str(baseline["policy_name"]),
            "internal test",
            baseline["internal_test_metrics"],
        ),
        _metric_row(
            str(learned["policy_name"]),
            "internal test",
            learned["internal_test_metrics"],
        ),
        "",
        "## Serving Status",
        "",
        f"- Serving ready: `{str(report['serving_ready']).lower()}`",
        "- Serving uses packaged article category metadata and does not read labels.",
        "- Candidate source position is the deterministic score tie-breaker.",
    ]
    if report.get("internal_holdout_warning"):
        lines.extend(
            [
                "",
                "## Holdout Warning",
                "",
                str(report["internal_holdout_warning"]),
            ]
        )
    return "\n".join(lines) + "\n"


def _best_candidate(candidates: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    parsed = [
        (name, summary)
        for name, summary in candidates.items()
        if isinstance(name, str) and isinstance(summary, dict)
    ]
    if not parsed:
        raise ValueError("No policy candidates were found")
    return sorted(
        parsed,
        key=lambda item: (
            -_sortable_metric(item[1], PRIMARY_METRIC),
            -_sortable_metric(item[1], "mrr"),
            -_sortable_metric(item[1], "auc"),
            item[0],
        ),
    )[0]


def _selected_learned_name(document: dict[str, Any]) -> str:
    selected = document.get("selected_model_name")
    if isinstance(selected, str) and selected:
        return selected
    return _best_candidate(_required_mapping(document, "rankers"))[0]


def _matching_summary(
    name: str,
    validation_summary: dict[str, Any],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    direct = candidates.get(name)
    if isinstance(direct, dict):
        return direct
    family = validation_summary.get("baseline_family") or validation_summary.get(
        "model_family"
    )
    matches = [
        summary
        for summary in candidates.values()
        if isinstance(summary, dict)
        and (
            summary.get("baseline_family") == family
            or summary.get("model_family") == family
        )
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Internal-test metrics are missing for candidate {name!r}")


def _metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    values = summary.get("metrics")
    if not isinstance(values, dict):
        raise ValueError("Candidate summary is missing metrics")
    metrics: dict[str, float | None] = {}
    for name, value in values.items():
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"Metric {name!r} must be numeric or null")
        metrics[str(name)] = None if value is None else float(value)
    return metrics


def _required_metric(
    metrics: dict[str, float | None],
    name: str,
    candidate_name: str,
) -> float:
    value = metrics.get(name)
    if value is None:
        raise ValueError(
            f"Candidate {candidate_name!r} is missing required metric {name!r}"
        )
    return float(value)


def _sortable_metric(summary: dict[str, Any], name: str) -> float:
    value = _metrics(summary).get(name)
    return float(value) if value is not None else float("-inf")


def _required_mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Metrics document is missing {key!r}")
    return value


def _policy_config(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config")
    return dict(config) if isinstance(config, dict) else {}


def _fitting_partitions(
    selected_family: str,
    validation_summary: dict[str, Any],
    test_summary: dict[str, Any],
) -> list[str]:
    del selected_family, validation_summary
    fit_metadata = test_summary.get("fit_metadata")
    if isinstance(fit_metadata, dict):
        values = fit_metadata.get("fitting_partitions")
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            return list(values)
    values = test_summary.get("fit_partitions")
    if isinstance(values, list) and all(isinstance(value, str) for value in values):
        return list(values)
    return ["train", "validation"]


def _write_news_catalog(processed_dir: Path, output_path: Path) -> None:
    news = load_news(processed_dir / PROCESSED_FILES["news"])
    rows = [
        {
            "news_id": item.news_id,
            "category": item.category,
            "subcategory": item.subcategory,
        }
        for item in sorted(news.values(), key=lambda item: item.news_id)
    ]
    schema = pa.schema(
        [
            pa.field("news_id", pa.string()),
            pa.field("category", pa.string()),
            pa.field("subcategory", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_path)


def _metric_row(name: str, split: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {split} | {_format_metric(metrics.get('ndcg@10'))} | "
        f"{_format_metric(metrics.get('mrr'))} | "
        f"{_format_metric(metrics.get('auc'))} |"
    )


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required policy-selection input is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in policy-selection input: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Policy-selection input must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None
