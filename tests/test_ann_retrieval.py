from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from feed_ranking_ops.data.prepare_dataset import prepare_dataset
from feed_ranking_ops.evaluation.processed import (
    BehaviorImpression,
    ImpressionCandidate,
    NewsItem,
)
from feed_ranking_ops.retrieval.ann_metrics import agreement_metrics, representation_loss_metrics
from feed_ranking_ops.retrieval.ann_protocol import render_ann_report, run_ann_benchmark
from feed_ranking_ops.retrieval.availability import ArticleAvailability, derive_article_availability
from feed_ranking_ops.retrieval.dense import (
    build_dense_user_profile,
    fingerprint_article_ids,
    fit_dense_article_index,
)
from feed_ranking_ops.retrieval.dense_exact import dense_exact_rank
from feed_ranking_ops.retrieval.faiss_backend import _post_filter_indices
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.run_ann_benchmark import build_parser
from feed_ranking_ops.retrieval.text import TextConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mindsmall_demo"


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
        ]
    )

    assert args.svd_dims == "32,64"
    assert args.top_k == 20


def test_ann_benchmark_workflow_writes_outputs_when_faiss_available(tmp_path: Path):
    pytest.importorskip("faiss")
    processed_dir = tmp_path / "processed"
    prepare_dataset(FIXTURE_DIR, processed_dir, tmp_path / "prepare_reports")
    reports_dir = tmp_path / "ann"

    result = run_ann_benchmark(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        svd_dims=[2],
        hnsw_m_values=[4],
        ef_construction_values=[8],
        ef_search_values=[8],
        oversampling_factors=[2],
        top_k=10,
        limit_queries=2,
        seed=42,
    )

    expected = [
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
    for filename in expected:
        assert (reports_dir / filename).is_file()
    assert result["protocol"]["test_labels_used_for_selection"] is False
    assert "configuration_name" in result["selected_configuration"]


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
