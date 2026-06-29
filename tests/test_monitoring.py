from pathlib import Path

from feed_ranking_ops.monitoring.diagnostics import (
    jensen_shannon_divergence,
)
from feed_ranking_ops.monitoring.report import generate_monitoring_report
from feed_ranking_ops.serving.schemas import RankRequestLogEvent

from test_policy_selection import _inputs, _select


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    result = _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.352,
        threshold=0.01,
    )
    del result
    return tmp_path / "processed", tmp_path / "serving"


def test_monitoring_report_runs_without_request_log(tmp_path: Path):
    processed_dir, serving_dir = _setup(tmp_path)

    result = generate_monitoring_report(
        processed_dir=processed_dir,
        serving_artifacts_dir=serving_dir,
        reports_dir=tmp_path / "monitoring",
        request_log=tmp_path / "missing.jsonl",
        limit_impressions=2,
    )

    report = result["report"]
    assert report["selected_policy"] == "category_affinity"
    assert report["request_log"]["available"] is False
    assert report["request_log"]["request_count"] == 0
    assert "validation" in report["offline_policy_diagnostics"]
    assert "test" in report["offline_policy_diagnostics"]
    assert "validation" in report["drift_checks"]
    assert (tmp_path / "monitoring" / "monitoring_report.json").is_file()
    assert (tmp_path / "monitoring" / "monitoring_report.md").is_file()


def test_monitoring_report_summarizes_synthetic_request_log(tmp_path: Path):
    processed_dir, serving_dir = _setup(tmp_path)
    request_log = tmp_path / "requests.jsonl"
    events = [
        _event(
            request_id="one",
            latency=2.0,
            history_count=0,
            candidate_count=3,
            ranked_count=3,
            missing=0,
            unknown=0,
            empty=True,
            categories={"news": 2, "sports": 1},
        ),
        _event(
            request_id="two",
            latency=8.0,
            history_count=2,
            candidate_count=2,
            ranked_count=1,
            missing=1,
            unknown=1,
            empty=False,
            categories={"sports": 1},
        ),
    ]
    request_log.write_text(
        "\n".join(event.model_dump_json() for event in events) + "\n",
        encoding="utf-8",
    )

    result = generate_monitoring_report(
        processed_dir=processed_dir,
        serving_artifacts_dir=serving_dir,
        reports_dir=tmp_path / "monitoring",
        request_log=request_log,
        limit_impressions=2,
    )
    summary = result["report"]["request_log"]

    assert summary["available"] is True
    assert summary["request_count"] == 2
    assert summary["latency_ms"]["mean"] == 5.0
    assert summary["latency_ms"]["p50"] == 5.0
    assert summary["missing_candidates"] == {"count": 1, "rate": 0.2}
    assert summary["unknown_history"] == {"count": 1, "rate": 0.5}
    assert summary["empty_history"] == {"count": 1, "rate": 0.5}
    assert summary["top_ranked_category_distribution"] == {
        "news": 0.5,
        "sports": 0.5,
    }
    assert "request_log" in result["report"]["drift_checks"]


def test_slice_diagnostics_include_expected_buckets_and_coverage(tmp_path: Path):
    processed_dir, serving_dir = _setup(tmp_path)

    report = generate_monitoring_report(
        processed_dir=processed_dir,
        serving_artifacts_dir=serving_dir,
        reports_dir=tmp_path / "monitoring",
        limit_impressions=2,
    )["report"]
    validation = report["offline_policy_diagnostics"]["validation"]

    assert validation["overall"]["n_impressions"] == 1
    assert validation["coverage"]["known_category_fraction"] == 1.0
    assert validation["coverage"]["known_subcategory_fraction"] == 1.0
    assert validation["score_ties"]["fraction"] >= 0.0
    assert validation["by_history_length"]
    assert validation["by_candidate_count"]
    assert validation["by_empty_history"]
    assert validation["by_top_ranked_category"]


def test_js_divergence_is_deterministic_and_bounded():
    identical = jensen_shannon_divergence(
        {"a": 0.5, "b": 0.5},
        {"a": 0.5, "b": 0.5},
    )
    disjoint = jensen_shannon_divergence({"a": 1.0}, {"b": 1.0})

    assert identical == 0.0
    assert disjoint == 1.0
    assert jensen_shannon_divergence({"b": 1.0}, {"a": 1.0}) == disjoint


def test_monitoring_report_is_deterministic(tmp_path: Path):
    processed_dir, serving_dir = _setup(tmp_path)

    first = generate_monitoring_report(
        processed_dir=processed_dir,
        serving_artifacts_dir=serving_dir,
        reports_dir=tmp_path / "first",
        limit_impressions=2,
    )
    second = generate_monitoring_report(
        processed_dir=processed_dir,
        serving_artifacts_dir=serving_dir,
        reports_dir=tmp_path / "second",
        limit_impressions=2,
    )

    assert first["report"] == second["report"]
    assert (
        tmp_path / "first" / "monitoring_report.json"
    ).read_text(encoding="utf-8") == (
        tmp_path / "second" / "monitoring_report.json"
    ).read_text(encoding="utf-8")


def test_invalid_monitoring_limit_is_rejected(tmp_path: Path):
    baseline_dir, ltr_dir, processed_dir = _inputs(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.352,
    )
    del baseline_dir, ltr_dir

    try:
        generate_monitoring_report(
            processed_dir=processed_dir,
            serving_artifacts_dir=tmp_path / "missing",
            reports_dir=tmp_path / "monitoring",
            limit_impressions=0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero monitoring limit should fail")


def _event(
    *,
    request_id: str,
    latency: float,
    history_count: int,
    candidate_count: int,
    ranked_count: int,
    missing: int,
    unknown: int,
    empty: bool,
    categories: dict[str, int],
) -> RankRequestLogEvent:
    return RankRequestLogEvent(
        timestamp="2026-01-01T00:00:00Z",
        request_id=request_id,
        selected_policy="category_affinity",
        history_id_count=history_count,
        candidate_id_count=candidate_count,
        ranked_candidate_count=ranked_count,
        missing_candidate_count=missing,
        unknown_history_count=unknown,
        empty_history=empty,
        latency_ms=latency,
        warnings=[],
        status="success",
        outcome="ranked",
        top_ranked_category_counts=categories,
    )
