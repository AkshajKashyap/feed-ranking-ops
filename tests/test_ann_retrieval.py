import csv
import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)
from feed_ranking_ops.retrieval import ann_protocol
from feed_ranking_ops.retrieval.ann_metrics import agreement_metrics, representation_loss_metrics
from feed_ranking_ops.retrieval.ann_protocol import (
    _runtime_environment,
    render_ann_report,
)
from feed_ranking_ops.retrieval.availability import ArticleAvailability, derive_article_availability
from feed_ranking_ops.retrieval.dense import (
    build_dense_user_profile,
    fingerprint_article_ids,
    fit_dense_article_index,
)
from feed_ranking_ops.retrieval.dense_exact import dense_exact_rank
from feed_ranking_ops.retrieval.faiss_backend import _post_filter_indices
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.run_ann_benchmark import build_parser, main as ann_main
from feed_ranking_ops.retrieval.text import TextConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"
ANN_OUTPUTS = [
    "validation_representation_metrics.json",
    "validation_ann_metrics.json",
    "test_representation_metrics.json",
    "test_ann_metrics.json",
    "config_sweep.csv",
    "latency_benchmark.csv",
    "model_comparison.md",
    "protocol.json",
    "runtime_environment.json",
    "selected_configuration.json",
    "index_metadata.json",
    "validation_retrievals.parquet",
    "test_retrievals.parquet",
    "query_diagnostics.parquet",
]
FAST_ANN_OUTPUTS = [
    "validation_metrics.json",
    "test_metrics.json",
    "config_sweep.csv",
    "latency_benchmark.csv",
    "model_comparison.md",
    "protocol.json",
    "runtime_environment.json",
    "selected_configuration.json",
    "index_metadata.json",
    "validation_retrievals.parquet",
    "test_retrievals.parquet",
    "query_diagnostics.parquet",
]


def _news() -> dict[str, NewsItem]:
    return {
        "A": NewsItem("A", "sports", "soccer", "apple match", "team wins"),
        "B": NewsItem("B", "sports", "soccer", "apple goal", "match recap"),
        "C": NewsItem("C", "finance", "markets", "bond market", "rates"),
        "D": NewsItem("D", "tech", "ai", "", ""),
    }


def _behavior(
    impression_id: str,
    *,
    hour: int,
    history: list[str],
    candidates: list[tuple[str, int | None]],
    partition: str = "train",
) -> BehaviorImpression:
    return BehaviorImpression(
        partition=partition,
        impression_id=impression_id,
        user_id="U",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        history_news_ids=history,
        candidates=[
            ImpressionCandidate(index, news_id, label)
            for index, (news_id, label) in enumerate(candidates)
        ],
    )


def _prepare_ann_smoke_fixture(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "prepare_reports")
    return processed_dir


def _run_ann_smoke_cli(processed_dir: Path, reports_dir: Path) -> None:
    exit_code = ann_main(
        [
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--limit-queries",
            "2",
            "--svd-dims",
            "2",
            "--hnsw-m",
            "4",
            "--ef-construction",
            "8",
            "--ef-search",
            "8",
            "--oversampling",
            "2",
            "--top-k",
            "10",
            "--seed",
            "42",
            "--faiss-threads",
            "1",
        ]
    )
    assert exit_code == 0


def _run_fast_ann_smoke_cli(processed_dir: Path, reports_dir: Path) -> None:
    exit_code = ann_main(
        [
            "--processed-dir",
            str(processed_dir),
            "--reports-dir",
            str(reports_dir),
            "--limit-queries",
            "2",
            "--svd-dims",
            "2",
            "--top-k",
            "10",
            "--faiss-threads",
            "1",
            "--ann-only",
            "--single-config",
        ]
    )
    assert exit_code == 0


def _without_timing_values(value):
    if isinstance(value, dict):
        return {
            key: _without_timing_values(item)
            for key, item in value.items()
            if key not in {"build_seconds", "efficiency"}
        }
    if isinstance(value, list):
        return [_without_timing_values(item) for item in value]
    return value


def test_dense_svd_dimension_is_capped_and_vectors_are_normalized():
    index = fit_dense_article_index(
        news=_news(),
        fitting_behaviors=[
            _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])
        ],
        text_config=TextConfig("title_abstract"),
        requested_dimension=256,
        seed=7,
        fitting_partitions=["train"],
    )

    assert index.requested_dimension == 256
    assert 1 <= index.effective_dimension <= 3
    assert index.vectors.shape == (4, index.effective_dimension)
    nonzero_rows = [
        row
        for article_id, row in index.article_to_row.items()
        if article_id not in index.zero_vector_article_ids
    ]
    norms = np.linalg.norm(index.vectors[nonzero_rows], axis=1)
    assert np.allclose(norms, 1.0)
    assert "D" in index.zero_vector_article_ids


def test_dense_svd_is_deterministic_for_same_seed():
    kwargs = {
        "news": _news(),
        "fitting_behaviors": [
            _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])
        ],
        "text_config": TextConfig("title_abstract"),
        "requested_dimension": 2,
        "seed": 42,
        "fitting_partitions": ["train"],
    }

    first = fit_dense_article_index(**kwargs)
    second = fit_dense_article_index(**kwargs)

    assert np.allclose(first.vectors, second.vectors)
    assert first.metadata()["article_fingerprint"] == second.metadata()["article_fingerprint"]


def test_dense_user_profile_matches_history_rules_and_normalizes():
    index = fit_dense_article_index(
        news=_news(),
        fitting_behaviors=[
            _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])
        ],
        text_config=TextConfig("title_abstract"),
        requested_dimension=2,
        seed=42,
        fitting_partitions=["train"],
    )

    profile = build_dense_user_profile(
        ["UNKNOWN", "A", "B"],
        index,
        HistoryProfileConfig("mean", max_history_length=2),
    )
    empty = build_dense_user_profile([], index, HistoryProfileConfig("mean"))

    assert profile.vector is not None
    assert profile.known_history_count == 2
    assert profile.unknown_history_count == 0
    assert np.linalg.norm(profile.vector) == pytest.approx(1.0)
    assert empty.fallback_reason == "empty_history"


def test_dense_exact_rank_uses_deterministic_tie_breaking():
    news = _news()
    fit = [_behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])]
    index = fit_dense_article_index(
        news=news,
        fitting_behaviors=fit,
        text_config=TextConfig("title_abstract"),
        requested_dimension=2,
        seed=42,
        fitting_partitions=["train"],
    )
    index.vectors[index.article_to_row["A"]] = np.array([1.0, 0.0], dtype=np.float32)
    index.vectors[index.article_to_row["B"]] = np.array([1.0, 0.0], dtype=np.float32)
    index.vectors[index.article_to_row["C"]] = np.array([0.0, 1.0], dtype=np.float32)
    availability = ArticleAvailability(
        first_candidate_timestamp={
            "B": datetime(2020, 1, 1, tzinfo=UTC),
            "A": datetime(2020, 1, 2, tzinfo=UTC),
            "C": datetime(2020, 1, 3, tzinfo=UTC),
        },
        first_history_timestamp={},
    )

    ranked, scores, diagnostics = dense_exact_rank(
        ["A", "B", "C"],
        dense_index=index,
        availability=availability,
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert ranked[:2] == ["B", "A"]
    assert scores["A"] == pytest.approx(scores["B"])
    assert diagnostics.candidates_scored == 3


def test_agreement_and_representation_metrics_are_explicit():
    agreement = agreement_metrics(["A", "B", "C"], ["A", "C", "D"], cutoffs=[2, 3])
    loss = representation_loss_metrics({"recall@100": 0.8}, {"recall@100": 0.6})

    assert agreement["top1_agreement"] == 1.0
    assert agreement["overlap_count@2"] == 1
    assert agreement["set_recall@3"] == pytest.approx(2 / 3)
    assert loss["dense_minus_sparse_recall@100"] == pytest.approx(-0.2)


def test_post_filter_counts_rejections_without_faiss_dependency():
    availability = ArticleAvailability(
        first_candidate_timestamp={"A": datetime(2020, 1, 1, tzinfo=UTC)},
        first_history_timestamp={},
    )

    filtered = _post_filter_indices(
        [0, 1, 1, 2, 99],
        article_ids=["A", "B", "C"],
        scores={0: 1.0, 1: 0.9, 2: 0.8, 99: 0.0},
        eligible_news_ids={"A", "B"},
        history_news_ids={"B"},
        availability=availability,
        exclude_history=True,
    )

    assert filtered.news_ids == ["A"]
    assert filtered.rejected_history_count == 1
    assert filtered.rejected_duplicate_count == 1
    assert filtered.rejected_unavailable_count == 1
    assert filtered.rejected_invalid_count == 1


def test_article_mapping_fingerprint_is_stable_and_order_sensitive():
    assert fingerprint_article_ids(["A", "B"]) == fingerprint_article_ids(["A", "B"])
    assert fingerprint_article_ids(["A", "B"]) != fingerprint_article_ids(["B", "A"])


def test_ann_report_contains_required_sections():
    minimal_metrics = {
        "metrics": {"recall@100": 1.0},
        "efficiency": {"p95_latency_seconds": 0.01},
    }
    representation_doc = {
        "sparse_exact": minimal_metrics,
        "dense_exact": minimal_metrics,
        "representation_loss": {"dense_minus_sparse_recall@100": 0.0},
    }
    ann_doc = {
        "selected_validation": {
            "clicked_target_metrics": minimal_metrics,
            "agreement_metrics": {"set_recall@100": 1.0},
            "index_memory_bytes": 32,
            "build_seconds": 0.01,
        }
    }

    report = render_ann_report(
        validation_representation_doc=representation_doc,
        validation_ann_doc=ann_doc,
        test_representation_doc=representation_doc,
        test_ann_doc={
            "faiss_flat": ann_doc["selected_validation"],
            "faiss_hnsw": ann_doc["selected_validation"],
        },
        protocol={
            "catalog_protocol": "observed_available",
            "top_k": 100,
            "smoke_test": True,
            "limit_queries": 2,
        },
        selected_doc={
            "configuration_name": "demo",
            "selection_basis": "validation ANN agreement only",
            "selection_metric": "agreement_set_recall@100",
        },
    )

    assert "Validation ANN Agreement" in report
    assert "Final Test ANN Quality" in report
    assert "representation loss" in report.lower()


def test_ann_cli_parser_imports_and_parses_options():
    args = build_parser().parse_args(
        [
            "--processed-dir",
            "data/processed",
            "--reports-dir",
            "reports/ann",
            "--svd-dims",
            "32,64",
            "--top-k",
            "20",
            "--ann-only",
            "--single-config",
            "--backend",
            "flat",
        ]
    )

    assert args.svd_dims == "32,64"
    assert args.top_k == 20
    assert args.ann_only is True
    assert args.single_config is True
    assert args.backend == "flat"


def test_ann_only_single_config_skips_dense_reference_and_writes_outputs(
    tmp_path: Path,
    monkeypatch,
):
    pytest.importorskip("faiss")
    processed_dir = _prepare_ann_smoke_fixture(tmp_path)
    reports_dir = tmp_path / "fast_ann"

    def fail_dense_reference(**kwargs):
        raise AssertionError("dense exact reference must be skipped in ANN-only mode")

    monkeypatch.setattr(ann_protocol, "_evaluate_sparse_dense", fail_dense_reference)
    _run_fast_ann_smoke_cli(processed_dir, reports_dir)

    for filename in FAST_ANN_OUTPUTS:
        assert (reports_dir / filename).is_file()
    protocol = json.loads((reports_dir / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (reports_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    test = json.loads((reports_dir / "test_metrics.json").read_text(encoding="utf-8"))
    with (reports_dir / "config_sweep.csv").open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 1

    assert protocol["ann_only"] is True
    assert protocol["single_config"] is True
    assert protocol["single_config_requested"] is True
    assert protocol["backend"] == "flat"
    assert protocol["dense_exact_comparison_skipped"] is True
    assert protocol["ann_approximation_recall_available"] is False
    assert protocol["timing"]["number_of_queries"] == 3
    assert protocol["timing"]["number_of_indexed_articles"] == 5
    assert protocol["timing"]["tfidf_vectorization_seconds"] >= 0
    assert protocol["timing"]["svd_fit_projection_seconds"] >= 0
    assert protocol["timing"]["faiss_index_build_seconds"] >= 0
    assert protocol["timing"]["faiss_search_seconds"] >= 0
    assert protocol["timing"]["total_runtime_seconds"] >= 0
    assert protocol["timing"]["peak_memory_bytes"] > 0
    assert validation["dense_exact_comparison_skipped"] is True
    assert validation["agreement_metrics"] is None
    assert validation["n_queries"] == 1
    assert test["n_queries"] == 2
    assert "Dense exact retrieval was skipped" in (
        reports_dir / "model_comparison.md"
    ).read_text(encoding="utf-8")
    for path in reports_dir.glob("*.parquet"):
        assert pq.read_table(path).num_rows > 0


def test_fast_ann_smoke_is_deterministic_when_faiss_available(tmp_path: Path):
    pytest.importorskip("faiss")
    processed_dir = _prepare_ann_smoke_fixture(tmp_path)
    first_dir = tmp_path / "first_fast"
    second_dir = tmp_path / "second_fast"

    _run_fast_ann_smoke_cli(processed_dir, first_dir)
    _run_fast_ann_smoke_cli(processed_dir, second_dir)

    for filename in ["validation_metrics.json", "test_metrics.json"]:
        first = json.loads((first_dir / filename).read_text(encoding="utf-8"))
        second = json.loads((second_dir / filename).read_text(encoding="utf-8"))
        assert first["metrics"] == second["metrics"]
        assert first["configuration"] == second["configuration"]
    for filename in ["validation_retrievals.parquet", "test_retrievals.parquet"]:
        columns = [
            "partition",
            "impression_id",
            "method",
            "retrieved_rank",
            "retrieved_news_id",
            "score",
        ]
        first = pq.read_table(first_dir / filename, columns=columns)
        second = pq.read_table(second_dir / filename, columns=columns)
        assert first.equals(second)


def test_ann_smoke_workflow_writes_readable_outputs_when_faiss_available(tmp_path: Path):
    pytest.importorskip("faiss")
    processed_dir = _prepare_ann_smoke_fixture(tmp_path)
    reports_dir = tmp_path / "ann"

    _run_ann_smoke_cli(processed_dir, reports_dir)

    for filename in ANN_OUTPUTS:
        assert (reports_dir / filename).is_file()
    for path in reports_dir.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))
    for path in reports_dir.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as source:
            assert list(csv.DictReader(source))
    for path in reports_dir.glob("*.parquet"):
        assert pq.read_table(path).num_rows > 0
    assert "# Dense Vector and FAISS ANN Retrieval Benchmark" in (
        reports_dir / "model_comparison.md"
    ).read_text(encoding="utf-8")


def test_ann_smoke_is_deterministic_when_faiss_available(tmp_path: Path):
    pytest.importorskip("faiss")
    processed_dir = _prepare_ann_smoke_fixture(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    _run_ann_smoke_cli(processed_dir, first_dir)
    _run_ann_smoke_cli(processed_dir, second_dir)

    for filename in [
        "validation_representation_metrics.json",
        "validation_ann_metrics.json",
        "test_representation_metrics.json",
        "test_ann_metrics.json",
        "selected_configuration.json",
    ]:
        first = json.loads((first_dir / filename).read_text(encoding="utf-8"))
        second = json.loads((second_dir / filename).read_text(encoding="utf-8"))
        assert _without_timing_values(first) == _without_timing_values(second)

    for filename in ["validation_retrievals.parquet", "test_retrievals.parquet"]:
        first = pq.read_table(first_dir / filename).select(
            ["partition", "impression_id", "method", "retrieved_rank", "retrieved_news_id"]
        )
        second = pq.read_table(second_dir / filename).select(
            ["partition", "impression_id", "method", "retrieved_rank", "retrieved_news_id"]
        )
        assert first.equals(second)


def test_ann_runtime_timestamp_is_timezone_aware_utc_without_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        timestamp = _runtime_environment()["created_at"]

    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert timestamp.endswith("Z")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_faiss_dynamic_availability_uses_only_eligible_articles(tmp_path: Path):
    pytest.importorskip("faiss")
    from feed_ranking_ops.retrieval.faiss_backend import FaissIndexConfig, build_faiss_index

    news = _news()
    fit = [
        _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)]),
    ]
    index = fit_dense_article_index(
        news=news,
        fitting_behaviors=fit,
        text_config=TextConfig("title_abstract"),
        requested_dimension=2,
        seed=42,
        fitting_partitions=["train"],
    )
    faiss_index = build_faiss_index(index, FaissIndexConfig(index_type="flat", oversampling_factor=1))
    availability = derive_article_availability({"train": fit})

    search = faiss_index.search(
        index.vectors[index.article_to_row["A"]],
        eligible_news_ids={"B"},
        history_news_ids=set(),
        availability=availability,
        top_k=1,
        exclude_history=False,
    )

    assert search.news_ids == ["B"]
    assert search.raw_search_calls >= 1
    assert search.rejected_unavailable_count >= 1
