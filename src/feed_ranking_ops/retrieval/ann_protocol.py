from __future__ import annotations

import csv
import json
import platform
import resource
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.processed import BehaviorImpression, ProcessedDataset, load_processed_dataset
from feed_ranking_ops.retrieval.ann_metrics import (
    aggregate_agreement_metrics,
    agreement_metrics,
    representation_loss_metrics,
)
from feed_ranking_ops.retrieval.availability import (
    CatalogEligibilityIndex,
    CatalogProtocol,
    build_catalog_eligibility_index,
    derive_article_availability,
    eligible_catalog,
    static_catalog_from_partitions,
    target_availability,
)
from feed_ranking_ops.retrieval.dense import DenseArticleIndex, build_dense_user_profile, fit_dense_article_index
from feed_ranking_ops.retrieval.dense_exact import retrieve_dense_exact_for_query
from feed_ranking_ops.retrieval.exact import RetrievedArticle, RetrievalResult, retrieve_for_query
from feed_ranking_ops.retrieval.faiss_backend import (
    FaissIndexConfig,
    LoadedFaissIndex,
    build_faiss_index,
    load_faiss_index,
    require_faiss,
    save_faiss_index,
    set_faiss_threads,
)
from feed_ranking_ops.retrieval.metrics import evaluate_retrieval_results
from feed_ranking_ops.retrieval.popularity import PopularityFallback, fit_popularity_fallback
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.queries import RetrievalQuery, behaviors_to_retrieval_queries
from feed_ranking_ops.retrieval.text import TextConfig

DEFAULT_SVD_DIMS = [32, 64, 128, 256]
DEFAULT_HNSW_M = [16]
DEFAULT_EF_CONSTRUCTION = [80]
DEFAULT_EF_SEARCH = [64, 128]
DEFAULT_OVERSAMPLING = [4, 8]
DEFAULT_TOP_K = 100


@dataclass(frozen=True)
class AnnBenchmarkConfiguration:
    requested_dimension: int
    index_type: str
    hnsw_m: int
    ef_construction: int
    ef_search: int
    oversampling_factor: int
    text_config: str = "title_abstract_category"
    profile_type: str = "mean"
    max_history_length: int | None = 50
    decay: float | None = None
    exclude_history: bool = True

    @property
    def name(self) -> str:
        history = "all" if self.max_history_length is None else str(self.max_history_length)
        return (
            f"svd={self.requested_dimension}__index={self.index_type}"
            f"__m={self.hnsw_m}__efc={self.ef_construction}"
            f"__efs={self.ef_search}__over={self.oversampling_factor}"
            f"__text={self.text_config}__profile={self.profile_type}"
            f"__history={history}"
        )

    def faiss_config(self) -> FaissIndexConfig:
        return FaissIndexConfig(
            index_type=self.index_type,  # type: ignore[arg-type]
            hnsw_m=self.hnsw_m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            oversampling_factor=self.oversampling_factor,
        )

    def profile_config(self) -> HistoryProfileConfig:
        return HistoryProfileConfig(
            profile_type=self.profile_type,  # type: ignore[arg-type]
            max_history_length=self.max_history_length,
            decay=self.decay,
        )


@dataclass(frozen=True)
class PreparedAnnQuery:
    query: RetrievalQuery
    eligible_news_ids: list[str]
    eligible_news_id_set: frozenset[str]
    history_news_ids: frozenset[str]
    target_info: dict[str, object]


def run_ann_benchmark(
    *,
    processed_dir: Path,
    reports_dir: Path,
    svd_dims: list[int] | None = None,
    hnsw_m_values: list[int] | None = None,
    ef_construction_values: list[int] | None = None,
    ef_search_values: list[int] | None = None,
    oversampling_factors: list[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
    limit_queries: int | None = None,
    catalog_protocol: CatalogProtocol = "observed_available",
    seed: int = 42,
    faiss_threads: int | None = None,
    save_index: bool = False,
    load_index: Path | None = None,
    ann_only: bool = False,
    single_config: bool = False,
    backend: str = "flat",
    text_config: str = "title",
    profile_method: str = "mean",
    max_history_length: int | None = None,
    profile_decay: float | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if limit_queries is not None and limit_queries <= 0:
        raise ValueError("limit_queries must be positive when provided")
    dimensions = _validate_positive_grid(svd_dims or DEFAULT_SVD_DIMS, "svd_dims")
    hnsw_m_values = _validate_positive_grid(hnsw_m_values or DEFAULT_HNSW_M, "hnsw_m_values")
    ef_construction_values = _validate_positive_grid(
        ef_construction_values or DEFAULT_EF_CONSTRUCTION,
        "ef_construction_values",
    )
    ef_search_values = _validate_positive_grid(ef_search_values or DEFAULT_EF_SEARCH, "ef_search_values")
    oversampling_factors = _validate_positive_grid(
        oversampling_factors or DEFAULT_OVERSAMPLING,
        "oversampling_factors",
    )
    if backend not in {"flat", "hnsw"}:
        raise ValueError("backend must be flat or hnsw")
    if profile_method not in {"mean", "recency"}:
        raise ValueError("profile_method must be mean or recency")
    if profile_method == "recency" and (
        profile_decay is None or not 0 < profile_decay <= 1
    ):
        raise ValueError("recency profile requires profile_decay in (0, 1]")
    if single_config:
        dimensions = dimensions[:1]
        hnsw_m_values = hnsw_m_values[:1]
        ef_construction_values = ef_construction_values[:1]
        ef_search_values = ef_search_values[:1]
        oversampling_factors = oversampling_factors[:1]
    set_faiss_threads(faiss_threads)
    require_faiss()

    if ann_only:
        return _run_ann_only_benchmark(
            processed_dir=processed_dir,
            reports_dir=reports_dir,
            requested_dimension=dimensions[0],
            hnsw_m=hnsw_m_values[0],
            ef_construction=ef_construction_values[0],
            ef_search=ef_search_values[0],
            oversampling_factor=oversampling_factors[0],
            backend=backend,
            text_config=text_config,
            profile_method=profile_method,
            max_history_length=max_history_length,
            profile_decay=profile_decay,
            top_k=top_k,
            limit_queries=limit_queries,
            catalog_protocol=catalog_protocol,
            seed=seed,
            faiss_threads=faiss_threads,
            save_index=save_index,
            single_config=single_config,
        )

    dataset = load_processed_dataset(processed_dir)
    availability = derive_article_availability(dataset.behaviors)
    train = _limit(dataset.behaviors["train"], limit_queries)
    validation = _limit(dataset.behaviors["validation"], limit_queries)
    test = _limit(dataset.behaviors["test"], limit_queries)
    validation_queries = behaviors_to_retrieval_queries(validation)
    test_queries = behaviors_to_retrieval_queries(test)

    validation_runs: list[dict[str, Any]] = []
    representation_by_dimension: dict[int, dict[str, Any]] = {}
    for dimension in dimensions:
        dense_index = fit_dense_article_index(
            news=dataset.news,
            fitting_behaviors=train,
            text_config=TextConfig("title_abstract_category"),
            requested_dimension=dimension,
            seed=seed,
            fitting_partitions=["train"],
        )
        fallback = fit_popularity_fallback(train, fitting_partitions=["train"])
        static_catalog = _static_catalog(catalog_protocol, train, validation, dataset)
        representation = _evaluate_sparse_dense(
            dataset=dataset,
            dense_index=dense_index,
            fallback=fallback,
            eval_queries=validation_queries,
            eval_behaviors=validation,
            availability=availability,
            catalog_protocol=catalog_protocol,
            static_catalog_ids=static_catalog,
            profile_config=HistoryProfileConfig("mean", max_history_length=50),
            top_k=top_k,
            partition_name="validation",
        )
        representation_by_dimension[dimension] = representation
        for hnsw_m, ef_construction, ef_search, oversampling in product(
            hnsw_m_values,
            ef_construction_values,
            ef_search_values,
            oversampling_factors,
        ):
            config = AnnBenchmarkConfiguration(
                requested_dimension=dimension,
                index_type="hnsw",
                hnsw_m=hnsw_m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                oversampling_factor=oversampling,
            )
            build_start = perf_counter()
            faiss_index = build_faiss_index(dense_index, config.faiss_config())
            build_seconds = perf_counter() - build_start
            ann_eval = _evaluate_faiss_method(
                method_name="faiss_hnsw",
                dataset=dataset,
                dense_index=dense_index,
                faiss_index=faiss_index,
                fallback=fallback,
                eval_queries=validation_queries,
                availability=availability,
                catalog_protocol=catalog_protocol,
                static_catalog_ids=static_catalog,
                profile_config=config.profile_config(),
                top_k=top_k,
                exclude_history=config.exclude_history,
                dense_reference_results=representation["dense_results"],
            )
            validation_runs.append(
                {
                    "configuration": config,
                    "dense_metadata": dense_index.metadata(),
                    "build_seconds": build_seconds,
                    "index_memory_bytes": faiss_index.memory_bytes,
                    **ann_eval,
                }
            )

    selected = _select_ann_configuration(validation_runs, top_k=top_k)
    selected_config: AnnBenchmarkConfiguration = selected["configuration"]
    selected_validation_representation = representation_by_dimension[
        selected_config.requested_dimension
    ]

    final_dense_index = fit_dense_article_index(
        news=dataset.news,
        fitting_behaviors=[*train, *validation],
        text_config=TextConfig(selected_config.text_config),
        requested_dimension=selected_config.requested_dimension,
        seed=seed,
        fitting_partitions=["train", "validation"],
    )
    final_fallback = fit_popularity_fallback(
        [*train, *validation],
        fitting_partitions=["train", "validation"],
    )
    final_static_catalog = _static_catalog(catalog_protocol, [*train, *validation], test, dataset)
    test_representation = _evaluate_sparse_dense(
        dataset=dataset,
        dense_index=final_dense_index,
        fallback=final_fallback,
        eval_queries=test_queries,
        eval_behaviors=test,
        availability=availability,
        catalog_protocol=catalog_protocol,
        static_catalog_ids=final_static_catalog,
        profile_config=selected_config.profile_config(),
        top_k=top_k,
        partition_name="test",
    )

    flat_build_start = perf_counter()
    flat_index = build_faiss_index(
        final_dense_index,
        FaissIndexConfig(index_type="flat", oversampling_factor=selected_config.oversampling_factor),
    )
    flat_build_seconds = perf_counter() - flat_build_start
    if load_index is not None:
        hnsw_index = load_faiss_index(
            index_path=load_index,
            metadata_path=load_index.with_suffix(".metadata.json"),
            expected_dimension=final_dense_index.effective_dimension,
            expected_article_fingerprint=final_dense_index.article_fingerprint,
        )
        hnsw_build_seconds = 0.0
    else:
        hnsw_build_start = perf_counter()
        hnsw_index = build_faiss_index(final_dense_index, selected_config.faiss_config())
        hnsw_build_seconds = perf_counter() - hnsw_build_start

    test_flat = _evaluate_faiss_method(
        method_name="faiss_flat",
        dataset=dataset,
        dense_index=final_dense_index,
        faiss_index=flat_index,
        fallback=final_fallback,
        eval_queries=test_queries,
        availability=availability,
        catalog_protocol=catalog_protocol,
        static_catalog_ids=final_static_catalog,
        profile_config=selected_config.profile_config(),
        top_k=top_k,
        exclude_history=selected_config.exclude_history,
        dense_reference_results=test_representation["dense_results"],
    )
    test_hnsw = _evaluate_faiss_method(
        method_name="faiss_hnsw",
        dataset=dataset,
        dense_index=final_dense_index,
        faiss_index=hnsw_index,
        fallback=final_fallback,
        eval_queries=test_queries,
        availability=availability,
        catalog_protocol=catalog_protocol,
        static_catalog_ids=final_static_catalog,
        profile_config=selected_config.profile_config(),
        top_k=top_k,
        exclude_history=selected_config.exclude_history,
        dense_reference_results=test_representation["dense_results"],
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation_representation_metrics": reports_dir / "validation_representation_metrics.json",
        "validation_ann_metrics": reports_dir / "validation_ann_metrics.json",
        "test_representation_metrics": reports_dir / "test_representation_metrics.json",
        "test_ann_metrics": reports_dir / "test_ann_metrics.json",
        "config_sweep": reports_dir / "config_sweep.csv",
        "latency_benchmark": reports_dir / "latency_benchmark.csv",
        "report": reports_dir / "model_comparison.md",
        "protocol": reports_dir / "protocol.json",
        "runtime_environment": reports_dir / "runtime_environment.json",
        "selected_configuration": reports_dir / "selected_configuration.json",
        "index_metadata": reports_dir / "index_metadata.json",
        "validation_retrievals": reports_dir / "validation_retrievals.parquet",
        "test_retrievals": reports_dir / "test_retrievals.parquet",
        "query_diagnostics": reports_dir / "query_diagnostics.parquet",
    }
    validation_representation_doc = _representation_document(
        selected_validation_representation,
        partition="validation",
        selected_config=selected_config,
    )
    validation_ann_doc = {
        "selected_configuration_name": selected_config.name,
        "selection_metric": f"agreement_set_recall@{min(100, top_k)}",
        "selected_validation": _ann_metric_document(selected),
        "test_labels_used_for_selection": False,
    }
    test_representation_doc = _representation_document(
        test_representation,
        partition="test",
        selected_config=selected_config,
    )
    test_ann_doc = {
        "dense_exact_metrics": test_representation["dense_metrics"],
        "faiss_flat": _ann_metric_document(
            {
                **test_flat,
                "configuration": AnnBenchmarkConfiguration(
                    requested_dimension=selected_config.requested_dimension,
                    index_type="flat",
                    hnsw_m=selected_config.hnsw_m,
                    ef_construction=selected_config.ef_construction,
                    ef_search=selected_config.ef_search,
                    oversampling_factor=selected_config.oversampling_factor,
                ),
                "build_seconds": flat_build_seconds,
                "index_memory_bytes": flat_index.memory_bytes,
            }
        ),
        "faiss_hnsw": _ann_metric_document(
            {
                **test_hnsw,
                "configuration": selected_config,
                "build_seconds": hnsw_build_seconds,
                "index_memory_bytes": hnsw_index.memory_bytes,
            }
        ),
        "ann_loss_note": (
            "ANN agreement is measured relative to dense exact retrieval. Representation "
            "loss is measured separately relative to sparse TF-IDF exact retrieval."
        ),
    }
    selected_doc = {
        "configuration": asdict(selected_config),
        "configuration_name": selected_config.name,
        "selection_basis": "validation ANN agreement only",
        "selection_metric": validation_ann_doc["selection_metric"],
        "selected_validation_metrics": selected["agreement_metrics"],
        "test_labels_used_for_selection": False,
    }
    protocol = {
        "evaluation_type": "dense_vector_ann_retrieval_benchmark",
        "catalog_protocol": catalog_protocol,
        "top_k": top_k,
        "smoke_test": limit_queries is not None,
        "limit_queries": limit_queries,
        "seed": seed,
        "validation_fit_partitions": ["train"],
        "validation_eval_partition": "validation",
        "test_fit_partitions": ["train", "validation"],
        "test_eval_partition": "test",
        "test_labels_used_for_selection": False,
        "comparison_chain": [
            "sparse_tfidf_exact",
            "dense_svd_exact",
            "faiss_flat_ip",
            "faiss_hnsw_ip",
        ],
        "validation_selection": (
            "Choose SVD dimension and HNSW search parameters on validation only using "
            "ANN agreement against dense exact retrieval as the primary metric."
        ),
    }
    runtime = _runtime_environment()

    _write_json(outputs["validation_representation_metrics"], validation_representation_doc)
    _write_json(outputs["validation_ann_metrics"], validation_ann_doc)
    _write_json(outputs["test_representation_metrics"], test_representation_doc)
    _write_json(outputs["test_ann_metrics"], test_ann_doc)
    _write_json(outputs["protocol"], protocol)
    _write_json(outputs["runtime_environment"], runtime)
    _write_json(outputs["selected_configuration"], selected_doc)
    _write_json(outputs["index_metadata"], hnsw_index.metadata)
    _write_config_sweep(outputs["config_sweep"], validation_runs, top_k=top_k)
    _write_latency_benchmark(
        outputs["latency_benchmark"],
        validation_runs=validation_runs,
        test_flat={**test_flat, "build_seconds": flat_build_seconds, "index_memory_bytes": flat_index.memory_bytes},
        test_hnsw={**test_hnsw, "build_seconds": hnsw_build_seconds, "index_memory_bytes": hnsw_index.memory_bytes},
    )
    _write_retrievals(
        [
            *selected_validation_representation["dense_prediction_rows"],
            *selected["prediction_rows"],
        ],
        outputs["validation_retrievals"],
    )
    _write_retrievals(
        [
            *test_representation["sparse_prediction_rows"],
            *test_representation["dense_prediction_rows"],
            *test_flat["prediction_rows"],
            *test_hnsw["prediction_rows"],
        ],
        outputs["test_retrievals"],
    )
    _write_query_diagnostics(
        [*selected["diagnostic_rows"], *test_flat["diagnostic_rows"], *test_hnsw["diagnostic_rows"]],
        outputs["query_diagnostics"],
    )
    outputs["report"].write_text(
        render_ann_report(
            validation_representation_doc=validation_representation_doc,
            validation_ann_doc=validation_ann_doc,
            test_representation_doc=test_representation_doc,
            test_ann_doc=test_ann_doc,
            protocol=protocol,
            selected_doc=selected_doc,
        ),
        encoding="utf-8",
    )
    if save_index:
        save_faiss_index(
            hnsw_index,
            index_path=reports_dir / "faiss_hnsw.index",
            metadata_path=reports_dir / "faiss_hnsw.metadata.json",
        )
    return {
        "validation_representation_metrics": validation_representation_doc,
        "validation_ann_metrics": validation_ann_doc,
        "test_representation_metrics": test_representation_doc,
        "test_ann_metrics": test_ann_doc,
        "selected_configuration": selected_doc,
        "protocol": protocol,
        "runtime_environment": runtime,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def _run_ann_only_benchmark(
    *,
    processed_dir: Path,
    reports_dir: Path,
    requested_dimension: int,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    oversampling_factor: int,
    backend: str,
    text_config: str,
    profile_method: str,
    max_history_length: int | None,
    profile_decay: float | None,
    top_k: int,
    limit_queries: int | None,
    catalog_protocol: CatalogProtocol,
    seed: int,
    faiss_threads: int | None,
    save_index: bool,
    single_config: bool,
) -> dict[str, Any]:
    total_start = perf_counter()
    dataset_load_start = perf_counter()
    dataset = load_processed_dataset(processed_dir)
    dataset_load_seconds = perf_counter() - dataset_load_start

    availability_start = perf_counter()
    availability = derive_article_availability(dataset.behaviors)
    availability_seconds = perf_counter() - availability_start

    train = _limit(dataset.behaviors["train"], limit_queries)
    validation = _limit(dataset.behaviors["validation"], limit_queries)
    test = _limit(dataset.behaviors["test"], limit_queries)
    query_start = perf_counter()
    validation_queries = behaviors_to_retrieval_queries(validation)
    test_queries = behaviors_to_retrieval_queries(test)
    query_construction_seconds = perf_counter() - query_start

    config = AnnBenchmarkConfiguration(
        requested_dimension=requested_dimension,
        index_type=backend,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        oversampling_factor=oversampling_factor,
        text_config=text_config,
        profile_type=profile_method,
        max_history_length=max_history_length,
        decay=profile_decay,
    )

    validation_static_catalog = _static_catalog(
        catalog_protocol,
        train,
        validation,
        dataset,
    )
    validation_catalog_index = build_catalog_eligibility_index(
        news=dataset.news,
        availability=availability,
        protocol=catalog_protocol,
        static_catalog_ids=validation_static_catalog,
    )
    query_preparation_start = perf_counter()
    prepared_validation = _prepare_ann_queries(
        validation_queries,
        news_ids=set(dataset.news),
        catalog_index=validation_catalog_index,
        exclude_history=config.exclude_history,
    )
    validation_query_preparation_seconds = perf_counter() - query_preparation_start

    validation_dense_index = fit_dense_article_index(
        news=dataset.news,
        fitting_behaviors=train,
        text_config=TextConfig(config.text_config),  # type: ignore[arg-type]
        requested_dimension=config.requested_dimension,
        seed=seed,
        fitting_partitions=["train"],
    )
    validation_fallback = fit_popularity_fallback(
        train,
        fitting_partitions=["train"],
    )
    validation_index_start = perf_counter()
    validation_faiss_index = build_faiss_index(
        validation_dense_index,
        config.faiss_config(),
    )
    validation_index_build_seconds = perf_counter() - validation_index_start
    validation_eval = _evaluate_faiss_method(
        method_name=f"faiss_{backend}",
        dataset=dataset,
        dense_index=validation_dense_index,
        faiss_index=validation_faiss_index,
        fallback=validation_fallback,
        eval_queries=validation_queries,
        availability=availability,
        catalog_protocol=catalog_protocol,
        static_catalog_ids=validation_static_catalog,
        profile_config=config.profile_config(),
        top_k=top_k,
        exclude_history=config.exclude_history,
        dense_reference_results=None,
        prepared_queries=prepared_validation,
    )

    final_static_catalog = _static_catalog(
        catalog_protocol,
        [*train, *validation],
        test,
        dataset,
    )
    final_catalog_index = build_catalog_eligibility_index(
        news=dataset.news,
        availability=availability,
        protocol=catalog_protocol,
        static_catalog_ids=final_static_catalog,
    )
    query_preparation_start = perf_counter()
    prepared_test = _prepare_ann_queries(
        test_queries,
        news_ids=set(dataset.news),
        catalog_index=final_catalog_index,
        exclude_history=config.exclude_history,
    )
    test_query_preparation_seconds = perf_counter() - query_preparation_start

    final_dense_index = fit_dense_article_index(
        news=dataset.news,
        fitting_behaviors=[*train, *validation],
        text_config=TextConfig(config.text_config),  # type: ignore[arg-type]
        requested_dimension=config.requested_dimension,
        seed=seed,
        fitting_partitions=["train", "validation"],
    )
    final_fallback = fit_popularity_fallback(
        [*train, *validation],
        fitting_partitions=["train", "validation"],
    )
    final_index_start = perf_counter()
    final_faiss_index = build_faiss_index(
        final_dense_index,
        config.faiss_config(),
    )
    final_index_build_seconds = perf_counter() - final_index_start
    test_eval = _evaluate_faiss_method(
        method_name=f"faiss_{backend}",
        dataset=dataset,
        dense_index=final_dense_index,
        faiss_index=final_faiss_index,
        fallback=final_fallback,
        eval_queries=test_queries,
        availability=availability,
        catalog_protocol=catalog_protocol,
        static_catalog_ids=final_static_catalog,
        profile_config=config.profile_config(),
        top_k=top_k,
        exclude_history=config.exclude_history,
        dense_reference_results=None,
        prepared_queries=prepared_test,
    )

    timing = _ann_only_timing(
        total_start=total_start,
        dataset_load_seconds=dataset_load_seconds,
        availability_seconds=availability_seconds,
        query_construction_seconds=query_construction_seconds,
        query_preparation_seconds=(
            validation_query_preparation_seconds + test_query_preparation_seconds
        ),
        validation_dense_index=validation_dense_index,
        final_dense_index=final_dense_index,
        validation_index_build_seconds=validation_index_build_seconds,
        final_index_build_seconds=final_index_build_seconds,
        validation_eval=validation_eval,
        test_eval=test_eval,
        validation_query_count=len(validation_queries),
        test_query_count=len(test_queries),
        article_count=len(final_dense_index.article_ids),
    )
    validation_doc = _ann_only_metrics_document(
        partition="validation",
        config=config,
        dense_index=validation_dense_index,
        evaluation=validation_eval,
        index_build_seconds=validation_index_build_seconds,
    )
    test_doc = _ann_only_metrics_document(
        partition="test",
        config=config,
        dense_index=final_dense_index,
        evaluation=test_eval,
        index_build_seconds=final_index_build_seconds,
    )
    selected_doc = {
        "configuration": asdict(config),
        "configuration_name": config.name,
        "selection_basis": "explicit single configuration; no validation sweep",
        "selection_metric": None,
        "selected_validation_metrics": validation_eval["metrics"]["metrics"],
        "test_labels_used_for_selection": False,
    }
    protocol = {
        "evaluation_type": "faiss_ann_only_retrieval_smoke",
        "mode": "ann_only",
        "ann_only": True,
        "single_config": True,
        "single_config_requested": single_config,
        "backend": backend,
        "catalog_protocol": catalog_protocol,
        "top_k": top_k,
        "smoke_test": limit_queries is not None,
        "limit_queries": limit_queries,
        "seed": seed,
        "faiss_threads": faiss_threads,
        "validation_fit_partitions": ["train"],
        "validation_eval_partition": "validation",
        "test_fit_partitions": ["train", "validation"],
        "test_eval_partition": "test",
        "test_labels_used_for_selection": False,
        "selected_configuration": asdict(config),
        "selected_configuration_name": config.name,
        "dense_exact_comparison_skipped": True,
        "ann_approximation_recall_available": False,
        "dense_exact_comparison_note": (
            "Dense exact retrieval was skipped. ANN agreement and approximation recall "
            "against dense exact are unavailable for this run."
        ),
        "timing": timing,
    }
    runtime = _runtime_environment()

    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation_metrics": reports_dir / "validation_metrics.json",
        "test_metrics": reports_dir / "test_metrics.json",
        "config_sweep": reports_dir / "config_sweep.csv",
        "latency_benchmark": reports_dir / "latency_benchmark.csv",
        "report": reports_dir / "model_comparison.md",
        "protocol": reports_dir / "protocol.json",
        "runtime_environment": reports_dir / "runtime_environment.json",
        "selected_configuration": reports_dir / "selected_configuration.json",
        "index_metadata": reports_dir / "index_metadata.json",
        "validation_retrievals": reports_dir / "validation_retrievals.parquet",
        "test_retrievals": reports_dir / "test_retrievals.parquet",
        "query_diagnostics": reports_dir / "query_diagnostics.parquet",
    }
    artifact_write_start = perf_counter()
    _write_json(outputs["validation_metrics"], validation_doc)
    _write_json(outputs["test_metrics"], test_doc)
    _write_json(outputs["protocol"], protocol)
    _write_json(outputs["runtime_environment"], runtime)
    _write_json(outputs["selected_configuration"], selected_doc)
    _write_json(outputs["index_metadata"], final_faiss_index.metadata)
    _write_config_sweep(
        outputs["config_sweep"],
        [
            {
                "configuration": config,
                "dense_metadata": validation_dense_index.metadata(),
                "build_seconds": validation_index_build_seconds,
                "index_memory_bytes": validation_faiss_index.memory_bytes,
                **validation_eval,
            }
        ],
        top_k=top_k,
    )
    _write_ann_only_latency(
        outputs["latency_benchmark"],
        backend=backend,
        validation_eval=validation_eval,
        test_eval=test_eval,
        validation_build_seconds=validation_index_build_seconds,
        test_build_seconds=final_index_build_seconds,
        validation_index_memory_bytes=validation_faiss_index.memory_bytes,
        test_index_memory_bytes=final_faiss_index.memory_bytes,
    )
    _write_retrievals(
        validation_eval["prediction_rows"],
        outputs["validation_retrievals"],
    )
    _write_retrievals(test_eval["prediction_rows"], outputs["test_retrievals"])
    _write_query_diagnostics(
        [*validation_eval["diagnostic_rows"], *test_eval["diagnostic_rows"]],
        outputs["query_diagnostics"],
    )
    report_write_start = perf_counter()
    outputs["report"].write_text(
        render_ann_only_report(
            validation_doc=validation_doc,
            test_doc=test_doc,
            protocol=protocol,
            selected_doc=selected_doc,
        ),
        encoding="utf-8",
    )
    timing["report_writing_seconds"] = perf_counter() - report_write_start
    if save_index:
        save_faiss_index(
            final_faiss_index,
            index_path=reports_dir / f"faiss_{backend}.index",
            metadata_path=reports_dir / f"faiss_{backend}.metadata.json",
        )
    timing["total_runtime_seconds"] = perf_counter() - total_start
    timing["peak_memory_bytes"] = _peak_memory_bytes()
    timing["artifact_writing_seconds"] = perf_counter() - artifact_write_start
    _write_json(outputs["protocol"], protocol)
    outputs["report"].write_text(
        render_ann_only_report(
            validation_doc=validation_doc,
            test_doc=test_doc,
            protocol=protocol,
            selected_doc=selected_doc,
        ),
        encoding="utf-8",
    )
    return {
        "validation_metrics": validation_doc,
        "test_metrics": test_doc,
        "selected_configuration": selected_doc,
        "protocol": protocol,
        "runtime_environment": runtime,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def render_ann_only_report(
    *,
    validation_doc: dict[str, Any],
    test_doc: dict[str, Any],
    protocol: dict[str, Any],
    selected_doc: dict[str, Any],
) -> str:
    lines = [
        "# Fast FAISS ANN Retrieval Smoke Report",
        "",
        "This run evaluates one dense representation through one FAISS backend. It does "
        "not compute sparse exact or dense exact reference retrieval.",
        "",
        "## Protocol",
        "",
        f"- Backend: `{protocol['backend']}`",
        f"- Catalog protocol: `{protocol['catalog_protocol']}`",
        f"- Top K: {protocol['top_k']}",
        f"- Validation queries: {protocol['timing']['validation_query_count']}",
        f"- Internal-test queries: {protocol['timing']['test_query_count']}",
        f"- Indexed articles: {protocol['timing']['number_of_indexed_articles']}",
        "- Validation representation is fit on train only.",
        "- Final representation is refit on train plus validation.",
        "- Test labels are not used for selection or fitting.",
        "",
        "## Selected Configuration",
        "",
        f"- Name: `{selected_doc['configuration_name']}`",
        f"- Basis: {selected_doc['selection_basis']}",
        "",
        "## Validation Retrieval Metrics",
        "",
        _clicked_metric_table(validation_doc),
        "",
        "## Internal-Test Retrieval Metrics",
        "",
        _clicked_metric_table(test_doc),
        "",
        "## Timing",
        "",
        *[
            f"- {name}: {_fmt(value)}"
            for name, value in sorted(protocol["timing"].items())
            if isinstance(value, int | float)
        ],
        "",
        "## Dense Exact Comparison",
        "",
        f"**Skipped:** {protocol['dense_exact_comparison_note']}",
        "",
        "## Limitations",
        "",
        "- ANN approximation recall is unavailable without a dense exact reference.",
        "- This internal chronological holdout is not an official MIND benchmark.",
        "- Offline clicked-target retrieval does not establish online recommendation quality.",
    ]
    return "\n".join(lines) + "\n"


def render_ann_report(
    *,
    validation_representation_doc: dict[str, Any],
    validation_ann_doc: dict[str, Any],
    test_representation_doc: dict[str, Any],
    test_ann_doc: dict[str, Any],
    protocol: dict[str, Any],
    selected_doc: dict[str, Any],
) -> str:
    lines = [
        "# Dense Vector and FAISS ANN Retrieval Benchmark",
        "",
        "This benchmark compares sparse exact retrieval, dense SVD exact retrieval, and "
        "FAISS approximate retrieval. It is not a second-stage ranking experiment.",
        "",
        "## Protocol",
        "",
        f"- Catalog protocol: `{protocol['catalog_protocol']}`",
        f"- Top K: {protocol['top_k']}",
        f"- Smoke-test limit: {protocol['limit_queries'] if protocol['smoke_test'] else 'none'}",
        "- Validation fitting uses train only; final test fitting uses train plus validation.",
        "- Test labels are not used for configuration selection.",
        "",
        "## Selected ANN Configuration",
        "",
        f"- Name: `{selected_doc['configuration_name']}`",
        f"- Selection basis: {selected_doc['selection_basis']}",
        f"- Selection metric: `{selected_doc['selection_metric']}`",
        "",
        "## Validation Representation Quality",
        "",
        _metric_section(validation_representation_doc),
        "",
        "## Validation ANN Agreement",
        "",
        _ann_section(validation_ann_doc["selected_validation"]),
        "",
        "## Final Test Representation Quality",
        "",
        _metric_section(test_representation_doc),
        "",
        "## Final Test ANN Quality",
        "",
        "### FAISS Flat",
        "",
        _ann_section(test_ann_doc["faiss_flat"]),
        "",
        "### FAISS HNSW",
        "",
        _ann_section(test_ann_doc["faiss_hnsw"]),
        "",
        "## Interpretation",
        "",
        "- Sparse exact to dense exact measures representation loss from SVD compression.",
        "- Dense exact to FAISS measures ANN search loss using identical dense vectors.",
        "- FAISS Flat should behave as a correctness check for dense inner-product search.",
        "- HNSW is approximate; disagreement is reported instead of hidden.",
    ]
    return "\n".join(lines) + "\n"


def _evaluate_sparse_dense(
    *,
    dataset: ProcessedDataset,
    dense_index: DenseArticleIndex,
    fallback: PopularityFallback,
    eval_queries: list[RetrievalQuery],
    eval_behaviors: list[BehaviorImpression],
    availability,
    catalog_protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None,
    profile_config: HistoryProfileConfig,
    top_k: int,
    partition_name: str,
) -> dict[str, Any]:
    del eval_behaviors
    sparse_results = []
    dense_results = []
    sparse_rows = []
    dense_rows = []
    for query in eval_queries:
        sparse_result = retrieve_for_query(
            query,
            news=dataset.news,
            article_index=dense_index.tfidf_index,
            availability=availability,
            fallback=fallback,
            profile_config=profile_config,
            catalog_protocol=catalog_protocol,
            static_catalog_ids=static_catalog_ids,
            top_k=top_k,
            exclude_history=True,
        )
        dense_result = retrieve_dense_exact_for_query(
            query,
            news=dataset.news,
            dense_index=dense_index,
            availability=availability,
            fallback=fallback,
            profile_config=profile_config,
            catalog_protocol=catalog_protocol,
            static_catalog_ids=static_catalog_ids,
            top_k=top_k,
            exclude_history=True,
        )
        sparse_results.append(sparse_result)
        dense_results.append(dense_result)
        sparse_rows.extend(_prediction_rows(sparse_result, method_name="sparse_exact", configuration_name="sparse_exact"))
        dense_rows.extend(_prediction_rows(dense_result, method_name="dense_exact", configuration_name="dense_exact"))
    sparse_metrics = evaluate_retrieval_results(sparse_results, catalog_size_total=len(dataset.news))
    dense_metrics = evaluate_retrieval_results(dense_results, catalog_size_total=len(dataset.news))
    return {
        "partition": partition_name,
        "dense_metadata": dense_index.metadata(),
        "sparse_results": sparse_results,
        "dense_results": dense_results,
        "sparse_metrics": sparse_metrics,
        "dense_metrics": dense_metrics,
        "representation_loss": representation_loss_metrics(
            sparse_metrics["metrics"],
            dense_metrics["metrics"],
        ),
        "sparse_prediction_rows": sparse_rows,
        "dense_prediction_rows": dense_rows,
    }


def _evaluate_faiss_method(
    *,
    method_name: str,
    dataset: ProcessedDataset,
    dense_index: DenseArticleIndex,
    faiss_index: LoadedFaissIndex,
    fallback: PopularityFallback,
    eval_queries: list[RetrievalQuery],
    availability,
    catalog_protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None,
    profile_config: HistoryProfileConfig,
    top_k: int,
    exclude_history: bool,
    dense_reference_results: list[RetrievalResult] | None,
    prepared_queries: dict[str, PreparedAnnQuery] | None = None,
) -> dict[str, Any]:
    evaluation_start = perf_counter()
    results = []
    prediction_rows = []
    diagnostic_rows = []
    agreement_rows = []
    reference_by_impression = (
        {
            result.query.impression_id: [item.news_id for item in result.retrieved]
            for result in dense_reference_results
        }
        if dense_reference_results is not None
        else None
    )
    search_seconds = 0.0
    for query in eval_queries:
        prepared_query = (
            prepared_queries.get(query.impression_id)
            if prepared_queries is not None
            else None
        )
        result, diagnostics = _retrieve_faiss_for_query(
            query,
            method_name=method_name,
            news=dataset.news,
            dense_index=dense_index,
            faiss_index=faiss_index,
            availability=availability,
            fallback=fallback,
            profile_config=profile_config,
            catalog_protocol=catalog_protocol,
            static_catalog_ids=static_catalog_ids,
            top_k=top_k,
            exclude_history=exclude_history,
            prepared_query=prepared_query,
        )
        search_seconds += float(diagnostics.get("search_latency_seconds", 0.0))
        candidate_ids = [item.news_id for item in result.retrieved]
        if reference_by_impression is not None:
            reference_ids = reference_by_impression.get(query.impression_id, [])
            agreement = agreement_metrics(reference_ids, candidate_ids)
            agreement_rows.append(agreement)
        else:
            agreement = {}
        diagnostic_rows.append(
            {
                "partition": query.partition,
                "impression_id": query.impression_id,
                "method": method_name,
                "raw_search_calls": diagnostics["raw_search_calls"],
                "raw_candidates_examined": diagnostics["raw_candidates_examined"],
                "rejected_history_count": diagnostics["rejected_history_count"],
                "rejected_unavailable_count": diagnostics["rejected_unavailable_count"],
                "rejected_invalid_count": diagnostics["rejected_invalid_count"],
                "rejected_duplicate_count": diagnostics["rejected_duplicate_count"],
                "unable_to_fill_top_k": diagnostics["unable_to_fill_top_k"],
                "oversampled_search": diagnostics["oversampled_search"],
                "agreement_set_recall@100": agreement.get("set_recall@100"),
                "agreement_top1": agreement.get("top1_agreement"),
                "first_differing_rank": agreement.get("first_differing_rank"),
            }
        )
        results.append(result)
        prediction_rows.extend(
            _prediction_rows(
                result,
                method_name=method_name,
                configuration_name=faiss_index.metadata["metadata_fingerprint"],
            )
        )
    metric_start = perf_counter()
    metrics = evaluate_retrieval_results(results, catalog_size_total=len(dataset.news))
    metric_evaluation_seconds = perf_counter() - metric_start
    return {
        "metrics": metrics,
        "agreement_metrics": aggregate_agreement_metrics(agreement_rows),
        "results": results,
        "prediction_rows": prediction_rows,
        "diagnostic_rows": diagnostic_rows,
        "timing": {
            "search_seconds": search_seconds,
            "metric_evaluation_seconds": metric_evaluation_seconds,
            "total_evaluation_seconds": perf_counter() - evaluation_start,
        },
    }


def _retrieve_faiss_for_query(
    query: RetrievalQuery,
    *,
    method_name: str,
    news,
    dense_index: DenseArticleIndex,
    faiss_index: LoadedFaissIndex,
    availability,
    fallback: PopularityFallback,
    profile_config: HistoryProfileConfig,
    catalog_protocol: CatalogProtocol,
    static_catalog_ids: set[str] | None,
    top_k: int,
    exclude_history: bool,
    prepared_query: PreparedAnnQuery | None = None,
) -> tuple[RetrievalResult, dict[str, Any]]:
    start = perf_counter()
    if prepared_query is None:
        target_info = target_availability(
            query,
            news=news,
            availability=availability,
            protocol=catalog_protocol,
            static_catalog_ids=static_catalog_ids,
        )
        eligible = eligible_catalog(
            query,
            news=news,
            availability=availability,
            protocol=catalog_protocol,
            static_catalog_ids=static_catalog_ids,
        )
        history = set(query.history_news_ids)
        if exclude_history:
            eligible = [news_id for news_id in eligible if news_id not in history]
        eligible_set = frozenset(eligible)
    else:
        target_info = prepared_query.target_info
        eligible = prepared_query.eligible_news_ids
        eligible_set = prepared_query.eligible_news_id_set
        history = prepared_query.history_news_ids
    profile = build_dense_user_profile(query.history_news_ids, dense_index, profile_config)
    fallback_reason = profile.fallback_reason
    fallback_used = fallback_reason is not None
    diagnostics: dict[str, Any] = {
        "method": method_name,
        "raw_search_calls": 0,
        "raw_candidates_examined": 0,
        "rejected_history_count": 0,
        "rejected_unavailable_count": 0,
        "rejected_invalid_count": 0,
        "rejected_duplicate_count": 0,
        "unable_to_fill_top_k": False,
        "oversampled_search": False,
        "search_latency_seconds": 0.0,
    }
    if not eligible:
        ranked_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        fallback_used = True
        fallback_reason = fallback_reason or "no_eligible_articles"
    elif fallback_used:
        ranked_ids = fallback.rank(eligible, availability, top_k=top_k)
        score_by_id = {news_id: fallback.score(news_id) for news_id in ranked_ids}
    else:
        search = faiss_index.search(
            profile.vector,
            eligible_news_ids=eligible_set,
            history_news_ids=history,
            availability=availability,
            top_k=top_k,
            exclude_history=exclude_history,
        )
        ranked_ids = search.news_ids
        score_by_id = search.scores
        diagnostics.update(asdict(search))
        diagnostics["search_latency_seconds"] = search.latency_seconds
    history_ids = history
    retrieved = [
        RetrievedArticle(
            news_id=news_id,
            rank=rank,
            score=float(score_by_id[news_id]),
            was_in_history=news_id in history_ids,
        )
        for rank, news_id in enumerate(ranked_ids[:top_k], start=1)
    ]
    result = RetrievalResult(
        query=query,
        retrieved=retrieved,
        target_news_ids=list(query.clicked_target_news_ids),
        available_target_news_ids=list(target_info["available_targets"]),
        unavailable_target_count=len(target_info["unavailable_targets"]),
        missing_target_metadata_count=int(target_info["missing_metadata_count"]),
        catalog_size=len(eligible),
        known_history_count=profile.known_history_count,
        unknown_history_count=profile.unknown_history_count,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        latency_seconds=perf_counter() - start,
    )
    return result, diagnostics


def _select_ann_configuration(runs: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    if not runs:
        raise ValueError("ANN configuration grid produced no runs")
    cutoff = min(100, top_k)

    def key(run: dict[str, Any]) -> tuple[float, float, float, int, str]:
        agreement = run["agreement_metrics"].get(f"set_recall@{cutoff}")
        clicked = run["metrics"]["metrics"].get(f"recall@{cutoff}")
        p95 = run["metrics"]["efficiency"].get("p95_latency_seconds")
        return (
            -1.0 if agreement is None else float(agreement),
            -1.0 if clicked is None else float(clicked),
            -999999.0 if p95 is None else -float(p95),
            -int(run["index_memory_bytes"]),
            run["configuration"].name,
        )

    return max(runs, key=key)


def _representation_document(
    representation: dict[str, Any],
    *,
    partition: str,
    selected_config: AnnBenchmarkConfiguration,
) -> dict[str, Any]:
    return {
        "partition": partition,
        "selected_configuration": asdict(selected_config),
        "dense_metadata": representation["dense_metadata"],
        "sparse_exact": representation["sparse_metrics"],
        "dense_exact": representation["dense_metrics"],
        "representation_loss": representation["representation_loss"],
    }


def _ann_metric_document(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration": asdict(run["configuration"]),
        "configuration_name": run["configuration"].name,
        "clicked_target_metrics": run["metrics"],
        "agreement_metrics": run["agreement_metrics"],
        "build_seconds": run.get("build_seconds"),
        "index_memory_bytes": run.get("index_memory_bytes"),
    }


def _prediction_rows(
    result: RetrievalResult,
    *,
    method_name: str,
    configuration_name: str,
) -> list[dict[str, Any]]:
    targets = set(result.available_target_news_ids)
    return [
        {
            "partition": result.query.partition,
            "impression_id": result.query.impression_id,
            "user_id": result.query.user_id,
            "query_timestamp": result.query.timestamp.isoformat(),
            "method": method_name,
            "retrieved_rank": item.rank,
            "retrieved_news_id": item.news_id,
            "score": item.score,
            "is_clicked_target": item.news_id in targets,
            "was_in_history": item.was_in_history,
            "catalog_size": result.catalog_size,
            "fallback_used": result.fallback_used,
            "selected_configuration": configuration_name,
        }
        for item in result.retrieved
    ]


def _write_retrievals(rows: list[dict[str, Any]], path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("partition", pa.string()),
            pa.field("impression_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("query_timestamp", pa.string()),
            pa.field("method", pa.string()),
            pa.field("retrieved_rank", pa.int32()),
            pa.field("retrieved_news_id", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("is_clicked_target", pa.bool_()),
            pa.field("was_in_history", pa.bool_()),
            pa.field("catalog_size", pa.int32()),
            pa.field("fallback_used", pa.bool_()),
            pa.field("selected_configuration", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_query_diagnostics(rows: list[dict[str, Any]], path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("partition", pa.string()),
            pa.field("impression_id", pa.string()),
            pa.field("method", pa.string()),
            pa.field("raw_search_calls", pa.int32()),
            pa.field("raw_candidates_examined", pa.int32()),
            pa.field("rejected_history_count", pa.int32()),
            pa.field("rejected_unavailable_count", pa.int32()),
            pa.field("rejected_invalid_count", pa.int32()),
            pa.field("rejected_duplicate_count", pa.int32()),
            pa.field("unable_to_fill_top_k", pa.bool_()),
            pa.field("oversampled_search", pa.bool_()),
            pa.field("agreement_set_recall@100", pa.float64()),
            pa.field("agreement_top1", pa.float64()),
            pa.field("first_differing_rank", pa.int32()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_config_sweep(path: Path, runs: list[dict[str, Any]], *, top_k: int) -> None:
    cutoff = min(100, top_k)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "configuration_name",
                "requested_dimension",
                "effective_dimension",
                f"agreement_set_recall@{cutoff}",
                f"clicked_recall@{cutoff}",
                "top1_agreement",
                "p95_latency_seconds",
                "index_memory_bytes",
                "build_seconds",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "configuration_name": run["configuration"].name,
                    "requested_dimension": run["configuration"].requested_dimension,
                    "effective_dimension": run["dense_metadata"]["effective_dimension"],
                    f"agreement_set_recall@{cutoff}": run["agreement_metrics"].get(
                        f"set_recall@{cutoff}"
                    ),
                    f"clicked_recall@{cutoff}": run["metrics"]["metrics"].get(f"recall@{cutoff}"),
                    "top1_agreement": run["agreement_metrics"].get("top1_agreement"),
                    "p95_latency_seconds": run["metrics"]["efficiency"].get("p95_latency_seconds"),
                    "index_memory_bytes": run["index_memory_bytes"],
                    "build_seconds": run["build_seconds"],
                }
            )


def _write_latency_benchmark(
    path: Path,
    *,
    validation_runs: list[dict[str, Any]],
    test_flat: dict[str, Any],
    test_hnsw: dict[str, Any],
) -> None:
    rows = []
    for run in validation_runs:
        rows.append(_latency_row("validation", "faiss_hnsw", run))
    rows.append(_latency_row("test", "faiss_flat", test_flat))
    rows.append(_latency_row("test", "faiss_hnsw", test_hnsw))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latency_row(partition: str, method: str, run: dict[str, Any]) -> dict[str, Any]:
    efficiency = run["metrics"]["efficiency"]
    diagnostics = run.get("diagnostic_rows", [])
    return {
        "partition": partition,
        "method": method,
        "query_count": efficiency["total_query_count"],
        "mean_latency_seconds": efficiency["mean_latency_seconds"],
        "p50_latency_seconds": efficiency["p50_latency_seconds"],
        "p95_latency_seconds": efficiency["p95_latency_seconds"],
        "p99_latency_seconds": efficiency["p99_latency_seconds"],
        "build_seconds": run.get("build_seconds"),
        "index_memory_bytes": run.get("index_memory_bytes"),
        "raw_search_calls": sum(int(row["raw_search_calls"]) for row in diagnostics),
        "mean_candidates_examined": (
            mean(int(row["raw_candidates_examined"]) for row in diagnostics)
            if diagnostics
            else None
        ),
        "unable_to_fill_top_k_rate": (
            mean(1.0 if row["unable_to_fill_top_k"] else 0.0 for row in diagnostics)
            if diagnostics
            else None
        ),
        "oversampled_search_rate": (
            mean(1.0 if row["oversampled_search"] else 0.0 for row in diagnostics)
            if diagnostics
            else None
        ),
    }


def _metric_section(document: dict[str, Any]) -> str:
    sparse = document["sparse_exact"]["metrics"]
    dense = document["dense_exact"]["metrics"]
    lines = [
        "| Metric | Sparse exact | Dense exact | Dense - sparse |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in sorted(set(sparse) | set(dense)):
        loss = document["representation_loss"].get(f"dense_minus_sparse_{metric}")
        lines.append(
            f"| {metric} | {_fmt(sparse.get(metric))} | {_fmt(dense.get(metric))} | {_fmt(loss)} |"
        )
    return "\n".join(lines)


def _ann_section(document: dict[str, Any]) -> str:
    clicked = document["clicked_target_metrics"]["metrics"]
    agreement = document["agreement_metrics"]
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        *[f"| clicked {key} | {_fmt(value)} |" for key, value in sorted(clicked.items())],
        *[f"| agreement {key} | {_fmt(value)} |" for key, value in sorted(agreement.items())],
        f"| index memory bytes | {document.get('index_memory_bytes')} |",
        f"| build seconds | {_fmt(document.get('build_seconds'))} |",
    ]
    return "\n".join(lines)


def _clicked_metric_table(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    return "\n".join(
        [
            "| Metric | Value |",
            "| --- | ---: |",
            *[
                f"| {name} | {_fmt(value)} |"
                for name, value in sorted(metrics.items())
            ],
            f"| valid queries | {document['n_valid_queries']} |",
            f"| mean catalog size | {_fmt(document['mean_catalog_size'])} |",
            f"| fallback query rate | {_fmt(document['fallback_query_rate'])} |",
        ]
    )


def _prepare_ann_queries(
    queries: list[RetrievalQuery],
    *,
    news_ids: set[str],
    catalog_index: CatalogEligibilityIndex,
    exclude_history: bool,
) -> dict[str, PreparedAnnQuery]:
    prepared: dict[str, PreparedAnnQuery] = {}
    for query in queries:
        history = frozenset(query.history_news_ids)
        eligible = catalog_index.eligible_ids(query.timestamp)
        if exclude_history:
            eligible = [news_id for news_id in eligible if news_id not in history]
        targets_with_metadata = [
            news_id for news_id in query.clicked_target_news_ids if news_id in news_ids
        ]
        available_targets = [
            news_id
            for news_id in targets_with_metadata
            if catalog_index.contains(news_id, query.timestamp)
        ]
        available_set = set(available_targets)
        prepared[query.impression_id] = PreparedAnnQuery(
            query=query,
            eligible_news_ids=eligible,
            eligible_news_id_set=frozenset(eligible),
            history_news_ids=history,
            target_info={
                "targets_with_metadata": targets_with_metadata,
                "available_targets": available_targets,
                "unavailable_targets": [
                    news_id
                    for news_id in targets_with_metadata
                    if news_id not in available_set
                ],
                "missing_metadata_count": len(query.clicked_target_news_ids)
                - len(targets_with_metadata),
            },
        )
    return prepared


def _ann_only_timing(
    *,
    total_start: float,
    dataset_load_seconds: float,
    availability_seconds: float,
    query_construction_seconds: float,
    query_preparation_seconds: float,
    validation_dense_index: DenseArticleIndex,
    final_dense_index: DenseArticleIndex,
    validation_index_build_seconds: float,
    final_index_build_seconds: float,
    validation_eval: dict[str, Any],
    test_eval: dict[str, Any],
    validation_query_count: int,
    test_query_count: int,
    article_count: int,
) -> dict[str, Any]:
    validation_build = validation_dense_index.build_timing
    test_build = final_dense_index.build_timing
    return {
        "dataset_load_seconds": dataset_load_seconds,
        "availability_construction_seconds": availability_seconds,
        "query_construction_seconds": query_construction_seconds,
        "query_eligibility_preparation_seconds": query_preparation_seconds,
        "tfidf_vectorization_seconds": (
            validation_build["tfidf_vectorization_seconds"]
            + test_build["tfidf_vectorization_seconds"]
        ),
        "svd_fit_projection_seconds": (
            validation_build["svd_fit_projection_seconds"]
            + test_build["svd_fit_projection_seconds"]
        ),
        "dense_normalization_seconds": (
            validation_build["normalization_seconds"]
            + test_build["normalization_seconds"]
        ),
        "dense_representation_seconds": (
            validation_build["total_dense_representation_seconds"]
            + test_build["total_dense_representation_seconds"]
        ),
        "faiss_index_build_seconds": (
            validation_index_build_seconds + final_index_build_seconds
        ),
        "validation_index_build_seconds": validation_index_build_seconds,
        "test_index_build_seconds": final_index_build_seconds,
        "faiss_search_seconds": (
            validation_eval["timing"]["search_seconds"]
            + test_eval["timing"]["search_seconds"]
        ),
        "validation_search_seconds": validation_eval["timing"]["search_seconds"],
        "test_search_seconds": test_eval["timing"]["search_seconds"],
        "evaluation_seconds": (
            validation_eval["timing"]["total_evaluation_seconds"]
            + test_eval["timing"]["total_evaluation_seconds"]
        ),
        "metric_evaluation_seconds": (
            validation_eval["timing"]["metric_evaluation_seconds"]
            + test_eval["timing"]["metric_evaluation_seconds"]
        ),
        "validation_query_count": validation_query_count,
        "test_query_count": test_query_count,
        "number_of_queries": validation_query_count + test_query_count,
        "number_of_indexed_articles": article_count,
        "average_eligible_articles_per_validation_query": _average_prepared_catalog_size(
            validation_eval
        ),
        "average_eligible_articles_per_test_query": _average_prepared_catalog_size(
            test_eval
        ),
        "total_runtime_seconds": perf_counter() - total_start,
        "peak_memory_bytes": _peak_memory_bytes(),
    }


def _average_prepared_catalog_size(evaluation: dict[str, Any]) -> float | None:
    results = evaluation["results"]
    return float(mean(result.catalog_size for result in results)) if results else None


def _ann_only_metrics_document(
    *,
    partition: str,
    config: AnnBenchmarkConfiguration,
    dense_index: DenseArticleIndex,
    evaluation: dict[str, Any],
    index_build_seconds: float,
) -> dict[str, Any]:
    return {
        "partition": partition,
        "configuration": asdict(config),
        "configuration_name": config.name,
        "backend": config.index_type,
        "dense_metadata": dense_index.metadata(),
        "dense_exact_comparison_skipped": True,
        "agreement_metrics": None,
        "index_build_seconds": index_build_seconds,
        "timing": evaluation["timing"],
        **evaluation["metrics"],
    }


def _write_ann_only_latency(
    path: Path,
    *,
    backend: str,
    validation_eval: dict[str, Any],
    test_eval: dict[str, Any],
    validation_build_seconds: float,
    test_build_seconds: float,
    validation_index_memory_bytes: int,
    test_index_memory_bytes: int,
) -> None:
    rows = [
        _latency_row(
            "validation",
            f"faiss_{backend}",
            {
                **validation_eval,
                "build_seconds": validation_build_seconds,
                "index_memory_bytes": validation_index_memory_bytes,
            },
        ),
        _latency_row(
            "test",
            f"faiss_{backend}",
            {
                **test_eval,
                "build_seconds": test_build_seconds,
                "index_memory_bytes": test_index_memory_bytes,
            },
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _static_catalog(
    catalog_protocol: CatalogProtocol,
    fit_behaviors: list[BehaviorImpression],
    eval_behaviors: list[BehaviorImpression],
    dataset: ProcessedDataset,
) -> set[str] | None:
    if catalog_protocol != "static_partition_catalog":
        return None
    return static_catalog_from_partitions([*fit_behaviors, *eval_behaviors], dataset.news)


def _limit(values: list[BehaviorImpression], limit: int | None) -> list[BehaviorImpression]:
    return values[:limit] if limit is not None else values


def _validate_positive_grid(values: list[int], name: str) -> list[int]:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _runtime_environment() -> dict[str, Any]:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError:
        faiss_version = None
    else:
        faiss_version = getattr(faiss, "__version__", None)
    return {
        "created_at": _utc_timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "faiss": faiss_version,
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _peak_memory_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024
