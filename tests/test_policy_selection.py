import json
from pathlib import Path

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.ranking.selection import select_and_package_policy

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


def _summary(
    ndcg: float,
    *,
    family_key: str,
    family: str,
    test_marker: float = 0.0,
) -> dict:
    return {
        family_key: family,
        "config": (
            {
                "category_weight": 1.0,
                "subcategory_weight": 0.5,
                "fallback_score": 0.0,
            }
            if family == "category_affinity"
            else {}
        ),
        "fit_metadata": {"fitting_partitions": ["train", "validation"]},
        "metrics": {
            "ndcg@10": ndcg,
            "mrr": ndcg - 0.05,
            "auc": ndcg + 0.1,
            "test_marker": test_marker,
        },
    }


def _inputs(
    tmp_path: Path,
    *,
    baseline_validation: float,
    learned_validation: float,
    baseline_test: float = 0.31,
    learned_test: float = 0.30,
) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    prepare_dataset(
        FIXTURE_DIR,
        processed_dir,
        tmp_path / "prepare_reports",
    )
    metadata_path = processed_dir / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["protocol"] = "train_only_chronological"
    metadata["final_partition_type"] = "internal_chronological_holdout"
    metadata["comparability_warning"] = "Synthetic internal holdout warning."
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_dir = tmp_path / "baseline"
    ltr_dir = tmp_path / "ltr"
    baseline_dir.mkdir()
    ltr_dir.mkdir()
    category_validation = _summary(
        baseline_validation,
        family_key="baseline_family",
        family="category_affinity",
    )
    original_validation = _summary(
        baseline_validation - 0.1,
        family_key="baseline_family",
        family="original_order",
    )
    category_test = _summary(
        baseline_test,
        family_key="baseline_family",
        family="category_affinity",
        test_marker=baseline_test,
    )
    original_test = _summary(
        baseline_test - 0.1,
        family_key="baseline_family",
        family="original_order",
    )
    _write_json(
        baseline_dir / "validation_metrics.json",
        {
            "baselines": {
                "category_affinity": category_validation,
                "original_order": original_validation,
            }
        },
    )
    _write_json(
        baseline_dir / "test_metrics.json",
        {
            "baselines": {
                "category_affinity": category_test,
                "original_order": original_test,
            }
        },
    )
    learned_name = "hist_gradient_boosting_learning_rate-0.1"
    learned_validation_summary = _summary(
        learned_validation,
        family_key="model_family",
        family="hist_gradient_boosting",
    )
    learned_validation_summary["fit_partitions"] = ["train"]
    learned_test_summary = _summary(
        learned_test,
        family_key="model_family",
        family="hist_gradient_boosting",
        test_marker=learned_test,
    )
    learned_test_summary["fit_partitions"] = ["train", "validation"]
    _write_json(
        ltr_dir / "validation_metrics.json",
        {
            "selected_model_name": learned_name,
            "rankers": {learned_name: learned_validation_summary},
        },
    )
    _write_json(
        ltr_dir / "test_metrics.json",
        {"rankers": {learned_name: learned_test_summary}},
    )
    return baseline_dir, ltr_dir, processed_dir


def _select(
    tmp_path: Path,
    *,
    baseline_validation: float,
    learned_validation: float,
    threshold: float,
    baseline_test: float = 0.31,
    learned_test: float = 0.30,
) -> dict:
    baseline_dir, ltr_dir, processed_dir = _inputs(
        tmp_path,
        baseline_validation=baseline_validation,
        learned_validation=learned_validation,
        baseline_test=baseline_test,
        learned_test=learned_test,
    )
    return select_and_package_policy(
        baseline_reports_dir=baseline_dir,
        ltr_reports_dir=ltr_dir,
        processed_dir=processed_dir,
        reports_dir=tmp_path / "selection_reports",
        artifacts_dir=tmp_path / "serving",
        minimum_relative_improvement=threshold,
        created_at="2026-01-01T00:00:00Z",
    )


def test_baseline_selected_when_learned_does_not_clear_threshold(tmp_path: Path):
    result = _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.352,
        threshold=0.01,
    )

    assert result["report"]["selected_policy_name"] == "category_affinity"
    assert result["report"]["promotion_decision"] == "promote_baseline_policy"
    assert (
        result["report"]["learned_promotion_result"]
        == "rejected_insufficient_improvement"
    )
    assert result["manifest"]["serving_ready"] is True


def test_learned_selected_when_validation_improvement_is_clear(tmp_path: Path):
    result = _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.37,
        threshold=0.01,
    )

    assert result["report"]["promotion_decision"] == "promote_learned_ranker"
    assert result["report"]["learned_promotion_result"] == "promoted"
    assert result["manifest"]["serving_ready"] is False


def test_internal_test_metrics_do_not_change_selection(tmp_path: Path):
    result = _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.37,
        threshold=0.01,
        baseline_test=0.99,
        learned_test=0.01,
    )

    assert result["report"]["promotion_decision"] == "promote_learned_ranker"
    assert result["report"]["promotion_rule"]["internal_test_used_for_selection"] is False
    assert result["report"]["learned_candidate"]["internal_test_metrics"][
        "test_marker"
    ] == 0.01


def test_tie_breaking_is_deterministic(tmp_path: Path):
    baseline_dir, ltr_dir, processed_dir = _inputs(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.20,
    )
    validation_path = baseline_dir / "validation_metrics.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["baselines"]["original_order"]["metrics"] = dict(
        validation["baselines"]["category_affinity"]["metrics"]
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = select_and_package_policy(
        baseline_reports_dir=baseline_dir,
        ltr_reports_dir=ltr_dir,
        processed_dir=processed_dir,
        reports_dir=tmp_path / "selection_reports",
        artifacts_dir=tmp_path / "serving",
        created_at="2026-01-01T00:00:00Z",
    )

    assert result["report"]["strongest_baseline"]["policy_name"] == "category_affinity"


def test_manifest_and_reports_are_self_contained_and_warn_on_holdout(tmp_path: Path):
    result = _select(
        tmp_path,
        baseline_validation=0.35,
        learned_validation=0.352,
        threshold=0.01,
    )
    artifacts_dir = tmp_path / "serving"
    reports_dir = tmp_path / "selection_reports"

    assert (artifacts_dir / "policy_manifest.json").is_file()
    assert (artifacts_dir / "news_catalog.parquet").is_file()
    assert (reports_dir / "promotion_report.json").is_file()
    assert (reports_dir / "promotion_report.md").is_file()
    assert result["manifest"]["artifact_paths"] == {
        "news_catalog": "news_catalog.parquet"
    }
    assert result["manifest"]["request_schema_version"] == 1
    assert result["manifest"]["news_catalog_columns"] == [
        "news_id",
        "category",
        "subcategory",
    ]
    assert (
        result["manifest"]["internal_holdout_warning"]
        == "Synthetic internal holdout warning."
    )
    assert result["manifest"]["fitting_partitions"] == ["train", "validation"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
