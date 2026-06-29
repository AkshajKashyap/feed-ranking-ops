import json
import os
import subprocess
from pathlib import Path

import pytest

from feed_ranking_ops import __version__
from feed_ranking_ops.cli import main as cli_main
from feed_ranking_ops.cli import project_info
from feed_ranking_ops.portfolio.report import generate_portfolio_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_report_generation_from_minimal_inputs(tmp_path: Path):
    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    _write_portfolio_inputs(reports, artifacts)

    result = generate_portfolio_report(
        reports_dir=reports,
        artifacts_dir=artifacts,
        output_dir=tmp_path / "portfolio",
    )
    summary = result["summary"]

    assert summary["availability"]["complete"] is True
    assert summary["project"]["version"] == "0.1.0"
    assert summary["data"]["split_counts"] == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    assert summary["logged_candidate_baseline"]["name"] == "category_affinity"
    assert summary["learning_to_rank"]["name"] == "hgb"
    assert summary["promotion"]["selected_policy"] == "category_affinity"
    assert summary["exact_retrieval_smoke"]["total_runtime_seconds"] == 12.0
    assert summary["ann_benchmark"]["total_runtime_seconds"] == 3.0
    markdown = (tmp_path / "portfolio" / "portfolio_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Results At A Glance" in markdown
    assert "train-only chronological holdout" in markdown


def test_portfolio_report_marks_missing_inputs_without_failing(tmp_path: Path):
    result = generate_portfolio_report(
        reports_dir=tmp_path / "missing_reports",
        artifacts_dir=tmp_path / "missing_artifacts",
        output_dir=tmp_path / "portfolio",
    )

    assert result["summary"]["availability"]["complete"] is False
    assert len(result["summary"]["availability"]["missing_inputs"]) == 13
    assert result["summary"]["logged_candidate_baseline"]["available"] is False
    assert "Unavailable Inputs" in (
        tmp_path / "portfolio" / "portfolio_summary.md"
    ).read_text(encoding="utf-8")


def test_package_version_and_cli_version(capsys):
    assert __version__ == "0.1.0"
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "feed-ranking-ops 0.1.0"


def test_project_info_command_with_and_without_manifest(
    tmp_path: Path,
    capsys,
):
    unavailable = project_info(tmp_path / "missing.json")
    assert unavailable["manifest_status"] == "unavailable"
    assert unavailable["selected_policy"] is None

    manifest_path = tmp_path / "policy_manifest.json"
    _write_json(manifest_path, _manifest())
    assert cli_main(["project-info", "--manifest", str(manifest_path)]) == 0
    output = capsys.readouterr().out

    assert "feed-ranking-ops 0.1.0" in output
    assert "Selected policy: category_affinity" in output
    assert "docs/model_card.md" in output


def test_demo_script_fails_clearly_without_prepared_data(tmp_path: Path):
    environment = dict(os.environ)
    environment["FEED_RANKING_OPS_ROOT"] = str(tmp_path)
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "generate_demo.sh")],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert "Demo requires prepared MIND data" in result.stderr
    assert "make prepare-data-train-only" in result.stderr


def test_release_documentation_and_checklist_exist():
    required = [
        "docs/architecture.md",
        "docs/model_card.md",
        "docs/experimental_methodology.md",
        "docs/release_checklist.md",
    ]
    for relative_path in required:
        assert (PROJECT_ROOT / relative_path).is_file()

    checklist = (PROJECT_ROOT / "docs/release_checklist.md").read_text(
        encoding="utf-8"
    )
    for command in (
        "ruff check . --no-cache",
        "pytest -q",
        "make check",
        "bash scripts/generate_demo.sh",
        "make smoke-monitor",
        "make project-info",
        "make release-check",
    ):
        assert command in checklist


def test_tracked_portfolio_snapshot_matches_verified_headlines():
    summary = json.loads(
        (
            PROJECT_ROOT / "reports/portfolio/portfolio_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["project"]["version"] == "0.1.0"
    assert summary["data"]["split_counts"] == {
        "train": 109875,
        "validation": 23545,
        "test": 23545,
    }
    assert summary["promotion"]["selected_policy"] == "category_affinity"
    assert (
        summary["logged_candidate_baseline"]["internal_test_metrics"]["ndcg@10"]
        == 0.33774822594802306
    )
    assert (
        summary["learning_to_rank"]["internal_test_metrics"]["ndcg@10"]
        == 0.33511296456954703
    )


def _write_portfolio_inputs(reports: Path, artifacts: Path) -> None:
    baseline_validation = {
        "partition_sizes": {"train": 70, "validation": 15},
        "baselines": {
            "category_affinity": {"metrics": _metrics(0.35)},
            "original_order": {"metrics": _metrics(0.20)},
        },
    }
    baseline_test = {
        "partition_sizes": {"train": 70, "validation": 15, "test": 15},
        "baselines": {
            "category_affinity": {"metrics": _metrics(0.33)},
        },
    }
    ltr_validation = {
        "selected_model_name": "hgb",
        "rankers": {"hgb": {"metrics": _metrics(0.36)}},
    }
    ltr_test = {"rankers": {"hgb": {"metrics": _metrics(0.32)}}}
    exact_protocol = {
        "timing": {"number_of_queries": 20, "total_runtime_seconds": 12.0}
    }
    ann_protocol = {
        "backend": "flat",
        "mode": "ann_only",
        "dense_exact_comparison_skipped": True,
        "timing": {
            "number_of_queries": 200,
            "total_runtime_seconds": 3.0,
            "faiss_search_seconds": 0.1,
        },
    }
    promotion = {
        "selected_policy_name": "category_affinity",
        "promotion_decision": "promote_baseline_policy",
        "learned_promotion_result": "rejected_insufficient_improvement",
        "observed_validation_improvement_relative": 0.028,
        "promotion_rule": {"minimum_relative_improvement": 0.03},
    }
    monitoring = {
        "request_log": {"request_count": 3},
        "offline_policy_diagnostics": {
            "validation": {
                "overall": {"metrics": _metrics(0.35)},
                "score_ties": {"fraction": 0.9},
            },
            "test": {
                "overall": {"metrics": _metrics(0.33)},
                "score_ties": {"fraction": 0.92},
            },
        },
    }
    values = {
        reports / "baselines" / "validation_metrics.json": baseline_validation,
        reports / "baselines" / "test_metrics.json": baseline_test,
        reports / "retrieval_smoke_100" / "protocol.json": exact_protocol,
        reports / "retrieval_smoke_100" / "validation_metrics.json": {
            "metrics": {"recall@100": 0.1}
        },
        reports / "retrieval_smoke_100" / "test_metrics.json": {
            "metrics": {"recall@100": 0.09}
        },
        reports / "ann_1000_flat" / "protocol.json": ann_protocol,
        reports / "ann_1000_flat" / "validation_metrics.json": {
            "metrics": {"recall@100": 0.08}
        },
        reports / "ann_1000_flat" / "test_metrics.json": {
            "metrics": {"recall@100": 0.07}
        },
        reports / "ltr" / "validation_metrics.json": ltr_validation,
        reports / "ltr" / "test_metrics.json": ltr_test,
        reports / "model_selection" / "promotion_report.json": promotion,
        reports / "monitoring" / "monitoring_report.json": monitoring,
        artifacts / "serving" / "policy_manifest.json": _manifest(),
    }
    for path, payload in values.items():
        _write_json(path, payload)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "request_schema_version": 1,
        "news_catalog_columns": ["news_id", "category", "subcategory"],
        "selected_policy_name": "category_affinity",
        "selected_policy_family": "category_affinity",
        "selected_metric": "ndcg@10",
        "validation_metrics": _metrics(0.35),
        "internal_test_metrics": _metrics(0.33),
        "promotion_decision": "promote_baseline_policy",
        "learned_promotion_result": "rejected_insufficient_improvement",
        "promotion_threshold_relative": 0.03,
        "observed_validation_improvement_relative": 0.028,
        "created_at": "2026-01-01T00:00:00Z",
        "git_commit": None,
        "data_protocol": "train_only_chronological",
        "final_partition_type": "internal_chronological_holdout",
        "internal_holdout_warning": "Internal holdout.",
        "fitting_partitions": ["train", "validation"],
        "policy_config": {},
        "artifact_paths": {"news_catalog": "news_catalog.parquet"},
        "serving_ready": True,
        "limitations": [],
    }


def _metrics(ndcg: float) -> dict[str, float]:
    return {"ndcg@10": ndcg, "mrr": ndcg - 0.05, "auc": ndcg + 0.1}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
