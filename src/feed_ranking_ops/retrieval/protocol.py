from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from feed_ranking_ops.evaluation.processed import BehaviorImpression, ProcessedDataset, load_processed_dataset
from feed_ranking_ops.retrieval.availability import (
    CatalogProtocol,
    derive_article_availability,
    eligible_catalog,
    static_catalog_from_partitions,
)
from feed_ranking_ops.retrieval.exact import retrieve_for_query, validate_retrieval_result
from feed_ranking_ops.retrieval.metrics import evaluate_retrieval_results, query_summary
from feed_ranking_ops.retrieval.popularity import fit_popularity_fallback
from feed_ranking_ops.retrieval.profiles import HistoryProfileConfig
from feed_ranking_ops.retrieval.queries import behaviors_to_retrieval_queries
from feed_ranking_ops.retrieval.text import TextConfig, fit_article_text_index, sparse_memory_bytes

DEFAULT_TOP_K = 100
DEFAULT_TEXT_CONFIGS = ["title", "title_abstract", "title_abstract_category"]
DEFAULT_HISTORY_LENGTHS = [10, 25, 50, None]
DEFAULT_DECAYS = [0.5, 0.8]


@dataclass(frozen=True)
class RetrievalConfiguration:
    text_config: str
    profile_type: str
    max_history_length: int | None
    decay: float | None
    exclude_history: bool

    @property
    def name(self) -> str:
        length = "all" if self.max_history_length is None else str(self.max_history_length)
        decay = "none" if self.decay is None else f"{self.decay:g}"
        return (
            f"text={self.text_config}__profile={self.profile_type}"
            f"__history={length}__decay={decay}"
            f"__exclude_history={str(self.exclude_history).lower()}"
        )


def run_exact_retrieval_protocol(
    *,
    processed_dir: Path,
    reports_dir: Path,
    catalog_protocol: CatalogProtocol = "observed_available",
    top_k: int = DEFAULT_TOP_K,
    limit_queries: int | None = None,
    text_configs: list[str] | None = None,
    history_lengths: list[int | None] | None = None,
    decay_values: list[float] | None = None,
    exclude_history: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    del seed
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if limit_queries is not None and limit_queries <= 0:
        raise ValueError("limit_queries must be positive when provided")
    dataset = load_processed_dataset(processed_dir)
    availability = derive_article_availability(dataset.behaviors)
    train = _limit(dataset.behaviors["train"], limit_queries)
    validation = _limit(dataset.behaviors["validation"], limit_queries)
    test = _limit(dataset.behaviors["test"], limit_queries)
    validation_queries = behaviors_to_retrieval_queries(validation)
    test_queries = behaviors_to_retrieval_queries(test)
    configurations = make_configuration_grid(
        text_configs=text_configs or DEFAULT_TEXT_CONFIGS,
        history_lengths=history_lengths or DEFAULT_HISTORY_LENGTHS,
        decay_values=decay_values or DEFAULT_DECAYS,
        exclude_history=exclude_history,
    )

    validation_runs = []
    for config in configurations:
        validation_runs.append(
            _evaluate_configuration(
                config,
                dataset=dataset,
                fit_behaviors=train,
                eval_behaviors=validation,
                eval_queries=validation_queries,
                fitting_partitions=["train"],
                availability=availability,
                catalog_protocol=catalog_protocol,
                top_k=top_k,
            )
        )
    selected = select_configuration(validation_runs)
    test_run = _evaluate_configuration(
        selected["configuration"],
        dataset=dataset,
        fit_behaviors=[*train, *validation],
        eval_behaviors=test,
        eval_queries=test_queries,
        fitting_partitions=["train", "validation"],
        availability=availability,
        catalog_protocol=catalog_protocol,
        top_k=top_k,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "validation_metrics": reports_dir / "validation_metrics.json",
        "test_metrics": reports_dir / "test_metrics.json",
        "config_sweep": reports_dir / "config_sweep.csv",
        "report": reports_dir / "model_comparison.md",
        "protocol": reports_dir / "protocol.json",
        "availability_summary": reports_dir / "availability_summary.json",
        "validation_retrievals": reports_dir / "validation_retrievals.parquet",
        "test_retrievals": reports_dir / "test_retrievals.parquet",
    }
    protocol = {
        "evaluation_type": "full_catalog_candidate_retrieval",
        "catalog_protocol": catalog_protocol,
        "catalog_protocol_note": _catalog_protocol_note(catalog_protocol),
        "top_k": top_k,
        "smoke_test": limit_queries is not None,
        "limit_queries": limit_queries,
        "selection_metric": "recall@100",
        "secondary_selection_metrics": ["ndcg@20", "recall@20"],
        "validation_fit_partitions": ["train"],
        "validation_eval_partition": "validation",
        "test_fit_partitions": ["train", "validation"],
        "test_eval_partition": "test",
        "test_labels_used_for_selection": False,
        "selected_configuration": asdict(selected["configuration"]),
        "selected_configuration_name": selected["configuration"].name,
        "text_fitting_policy": (
            "Strict inductive: fit TF-IDF vocabulary on articles referenced by fitting "
            "partition histories or candidates; transform evaluation articles without "
            "fitting on evaluation labels."
        ),
    }
    availability_summary = {
        "article_count_with_candidate_availability": len(
            availability.first_candidate_timestamp
        ),
        "article_count_with_history_availability": len(availability.first_history_timestamp),
        "candidate_availability_is_publication_time": False,
        "note": (
            "Observed availability is the first behavior timestamp where an article appears "
            "as an impression candidate. It is not true publication time."
        ),
    }
    _write_json(output_paths["validation_metrics"], selected["metrics_document"])
    _write_json(output_paths["test_metrics"], test_run["metrics_document"])
    _write_json(output_paths["protocol"], protocol)
    _write_json(output_paths["availability_summary"], availability_summary)
    _write_config_sweep(output_paths["config_sweep"], validation_runs)
    _write_retrievals(selected["prediction_rows"], output_paths["validation_retrievals"])
    _write_retrievals(test_run["prediction_rows"], output_paths["test_retrievals"])
    output_paths["report"].write_text(
        render_retrieval_report(
            selected["metrics_document"],
            test_run["metrics_document"],
            protocol,
            availability_summary,
        ),
        encoding="utf-8",
    )
    return {
        "validation_metrics": selected["metrics_document"],
        "test_metrics": test_run["metrics_document"],
        "protocol": protocol,
        "availability_summary": availability_summary,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def make_configuration_grid(
    *,
    text_configs: list[str],
    history_lengths: list[int | None],
    decay_values: list[float],
    exclude_history: bool,
) -> list[RetrievalConfiguration]:
    grid: list[RetrievalConfiguration] = []
    for text_config in text_configs:
        for max_history_length in history_lengths:
            grid.append(
                RetrievalConfiguration(
                    text_config=text_config,
                    profile_type="mean",
                    max_history_length=max_history_length,
                    decay=None,
                    exclude_history=exclude_history,
                )
            )
            for decay in decay_values:
                grid.append(
                    RetrievalConfiguration(
                        text_config=text_config,
                        profile_type="recency",
                        max_history_length=max_history_length,
                        decay=decay,
                        exclude_history=exclude_history,
                    )
                )
    return grid


def select_configuration(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def key(run: dict[str, Any]) -> tuple[float, float, float, str]:
        metrics = run["metrics_document"]["metrics"]
        return (
            float(metrics.get("recall@100") or -1.0),
            float(metrics.get("ndcg@20") or -1.0),
            float(metrics.get("recall@20") or -1.0),
            run["configuration"].name,
        )

    return max(runs, key=key)


def render_retrieval_report(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    protocol: dict[str, Any],
    availability_summary: dict[str, Any],
) -> str:
    lines = [
        "# Exact Full-Catalog Retrieval Report",
        "",
        "This report evaluates candidate retrieval from an eligible article catalog. It is "
        "not logged-candidate ranking.",
        "",
        "## Protocol",
        "",
        f"- Catalog protocol: `{protocol['catalog_protocol']}`",
        f"- Catalog note: {protocol['catalog_protocol_note']}",
        "- Validation: fit TF-IDF and fallback popularity on train only.",
        "- Selection: choose configuration by validation Recall@100 with NDCG@20 and Recall@20 tie-breakers.",
        "- Final test: refit on train plus validation, then evaluate once on test.",
        f"- Smoke-test limit: {protocol['limit_queries'] if protocol['smoke_test'] else 'none'}",
        "",
        "## Selected Configuration",
        "",
        f"- Name: `{protocol['selected_configuration_name']}`",
        f"- Details: `{protocol['selected_configuration']}`",
        "",
        "## Availability",
        "",
        f"- Articles with candidate availability: {availability_summary['article_count_with_candidate_availability']}",
        f"- Articles with history-only diagnostic timestamps: {availability_summary['article_count_with_history_availability']}",
        "- Observed availability is not publication time.",
        "",
        "## Validation Metrics",
        "",
        _metrics_table(validation_metrics),
        "",
        "## Final Test Metrics",
        "",
        _metrics_table(test_metrics),
        "",
        "## Efficiency",
        "",
        f"- Validation efficiency: `{validation_metrics['efficiency']}`",
        f"- Test efficiency: `{test_metrics['efficiency']}`",
        "",
        "## Limitations",
        "",
        "- Exact retrieval is a correctness reference and may not scale to large catalogs.",
        "- Offline recall over clicked targets does not prove online engagement improvement.",
        "- Exposure bias remains because targets come from logged impressions.",
    ]
    return "\n".join(lines) + "\n"


def _evaluate_configuration(
    config: RetrievalConfiguration,
    *,
    dataset: ProcessedDataset,
    fit_behaviors: list[BehaviorImpression],
    eval_behaviors: list[BehaviorImpression],
    eval_queries,
    fitting_partitions: list[str],
    availability,
    catalog_protocol: CatalogProtocol,
    top_k: int,
) -> dict[str, Any]:
    article_index = fit_article_text_index(
        news=dataset.news,
        fitting_behaviors=fit_behaviors,
        text_config=TextConfig(config.text_config),
    )
    fallback = fit_popularity_fallback(
        fit_behaviors,
        fitting_partitions=fitting_partitions,
    )
    static_catalog = (
        static_catalog_from_partitions([*fit_behaviors, *eval_behaviors], dataset.news)
        if catalog_protocol == "static_partition_catalog"
        else None
    )
    profile_config = HistoryProfileConfig(
        profile_type=config.profile_type,  # type: ignore[arg-type]
        max_history_length=config.max_history_length,
        decay=config.decay,
    )
    results = []
    prediction_rows = []
    query_rows = []
    for query in eval_queries:
        result = retrieve_for_query(
            query,
            news=dataset.news,
            article_index=article_index,
            availability=availability,
            fallback=fallback,
            profile_config=profile_config,
            catalog_protocol=catalog_protocol,
            static_catalog_ids=static_catalog,
            top_k=top_k,
            exclude_history=config.exclude_history,
        )
        eligible = set(
            eligible_catalog(
                query,
                news=dataset.news,
                availability=availability,
                protocol=catalog_protocol,
                static_catalog_ids=static_catalog,
            )
        )
        if config.exclude_history:
            eligible = eligible.difference(query.history_news_ids)
        validate_retrieval_result(result, eligible_news_ids=eligible, exclude_history=config.exclude_history)
        results.append(result)
        prediction_rows.extend(_prediction_rows(result, config.name))
        query_rows.append(query_summary(result))
    metrics = evaluate_retrieval_results(results, catalog_size_total=len(dataset.news))
    metrics_document = {
        "configuration": asdict(config),
        "configuration_name": config.name,
        "catalog_protocol": catalog_protocol,
        "fit_metadata": {
            "fitting_partitions": list(fitting_partitions),
            "tfidf_fitting_article_count": len(article_index.fitting_article_ids),
            "tfidf_vocabulary_size": article_index.vocabulary_size,
            "fallback": fallback.metadata(),
            "approx_sparse_matrix_memory_bytes": sparse_memory_bytes(
                article_index.article_matrix
            ),
        },
        "query_summaries": query_rows,
        **metrics,
    }
    return {
        "configuration": config,
        "metrics_document": metrics_document,
        "prediction_rows": prediction_rows,
    }


def _prediction_rows(result, configuration_name: str) -> list[dict[str, Any]]:
    targets = set(result.available_target_news_ids)
    return [
        {
            "partition": result.query.partition,
            "impression_id": result.query.impression_id,
            "user_id": result.query.user_id,
            "query_timestamp": result.query.timestamp,
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
            pa.field("query_timestamp", pa.timestamp("us", tz="UTC")),
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


def _write_config_sweep(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "configuration_name",
                "recall@100",
                "ndcg@20",
                "recall@20",
                "valid_queries",
                "fallback_query_rate",
            ],
        )
        writer.writeheader()
        for run in runs:
            doc = run["metrics_document"]
            writer.writerow(
                {
                    "configuration_name": run["configuration"].name,
                    "recall@100": doc["metrics"].get("recall@100"),
                    "ndcg@20": doc["metrics"].get("ndcg@20"),
                    "recall@20": doc["metrics"].get("recall@20"),
                    "valid_queries": doc["n_valid_queries"],
                    "fallback_query_rate": doc["fallback_query_rate"],
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _metrics_table(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    return "\n".join(
        [
            "| Metric | Value |",
            "| --- | ---: |",
            *[
                f"| {name} | {_format_metric(value)} |"
                for name, value in sorted(metrics.items())
            ],
            f"| valid queries | {document['n_valid_queries']} |",
            f"| mean catalog size | {_format_metric(document['mean_catalog_size'])} |",
            f"| fallback query rate | {_format_metric(document['fallback_query_rate'])} |",
        ]
    )


def _format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _limit(values: list[BehaviorImpression], limit: int | None) -> list[BehaviorImpression]:
    return values[:limit] if limit is not None else values


def _catalog_protocol_note(protocol: CatalogProtocol) -> str:
    if protocol == "observed_available":
        return (
            "Default leakage-aware catalog: first observed candidate timestamp must be "
            "less than or equal to query timestamp."
        )
    return (
        "Diagnostic static catalog using all articles observed in fitting and evaluation "
        "partitions; this can be optimistic."
    )
