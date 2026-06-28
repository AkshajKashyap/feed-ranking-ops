from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from feed_ranking_ops.evaluation.processed import BehaviorImpression, ImpressionCandidate, NewsItem
from feed_ranking_ops.retrieval.availability import derive_article_availability
from feed_ranking_ops.retrieval.dense import fit_dense_article_index
from feed_ranking_ops.retrieval.faiss_backend import (
    FaissIndexConfig,
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
)
from feed_ranking_ops.retrieval.text import TextConfig

pytest.importorskip("faiss")


def _news() -> dict[str, NewsItem]:
    return {
        "A": NewsItem("A", "sports", "soccer", "apple match", "team wins"),
        "B": NewsItem("B", "sports", "soccer", "apple goal", "match recap"),
        "C": NewsItem("C", "finance", "markets", "bond market", "rates"),
    }


def _behavior(
    impression_id: str,
    *,
    hour: int,
    history: list[str],
    candidates: list[tuple[str, int | None]],
) -> BehaviorImpression:
    return BehaviorImpression(
        partition="train",
        impression_id=impression_id,
        user_id="U",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        history_news_ids=history,
        candidates=[
            ImpressionCandidate(index, news_id, label)
            for index, (news_id, label) in enumerate(candidates)
        ],
    )


def _dense_index():
    return fit_dense_article_index(
        news=_news(),
        fitting_behaviors=[
            _behavior("fit", hour=1, history=["A"], candidates=[("A", 1), ("B", 0), ("C", 0)])
        ],
        text_config=TextConfig("title_abstract"),
        requested_dimension=2,
        seed=42,
        fitting_partitions=["train"],
    )


def test_faiss_flat_matches_self_query_and_saves_roundtrip(tmp_path: Path):
    dense_index = _dense_index()
    loaded = build_faiss_index(dense_index, FaissIndexConfig(index_type="flat", oversampling_factor=2))
    availability = derive_article_availability(
        {
            "train": [
                _behavior("fit", hour=1, history=[], candidates=[("A", 1), ("B", 0), ("C", 0)])
            ]
        }
    )

    search = loaded.search(
        dense_index.vectors[dense_index.article_to_row["A"]],
        eligible_news_ids=set(dense_index.article_ids),
        history_news_ids=set(),
        availability=availability,
        top_k=1,
        exclude_history=False,
    )

    assert search.news_ids == ["A"]

    index_path = tmp_path / "flat.index"
    metadata_path = tmp_path / "flat.metadata.json"
    save_faiss_index(loaded, index_path=index_path, metadata_path=metadata_path)
    reloaded = load_faiss_index(
        index_path=index_path,
        metadata_path=metadata_path,
        expected_dimension=dense_index.effective_dimension,
        expected_article_fingerprint=dense_index.article_fingerprint,
    )

    assert reloaded.dimension == loaded.dimension
    assert reloaded.article_ids == loaded.article_ids


def test_faiss_hnsw_builds_and_searches():
    dense_index = _dense_index()
    loaded = build_faiss_index(
        dense_index,
        FaissIndexConfig(
            index_type="hnsw",
            hnsw_m=4,
            ef_construction=8,
            ef_search=8,
            oversampling_factor=2,
        ),
    )
    availability = derive_article_availability(
        {
            "train": [
                _behavior("fit", hour=1, history=[], candidates=[("A", 1), ("B", 0), ("C", 0)])
            ]
        }
    )

    search = loaded.search(
        dense_index.vectors[dense_index.article_to_row["A"]],
        eligible_news_ids=set(dense_index.article_ids),
        history_news_ids=set(),
        availability=availability,
        top_k=2,
        exclude_history=False,
    )

    assert len(search.news_ids) == 2
    assert search.raw_candidates_examined >= 2


def test_faiss_batch_search_respects_eligibility_and_history():
    dense_index = _dense_index()
    loaded = build_faiss_index(
        dense_index,
        FaissIndexConfig(index_type="flat", oversampling_factor=1),
    )
    availability = derive_article_availability(
        {
            "train": [
                _behavior(
                    "fit",
                    hour=1,
                    history=[],
                    candidates=[("A", 1), ("B", 0), ("C", 0)],
                )
            ]
        }
    )
    vectors = np.vstack(
        [
            dense_index.vectors[dense_index.article_to_row["A"]],
            dense_index.vectors[dense_index.article_to_row["A"]],
        ]
    )

    batch = loaded.search_batch(
        vectors,
        eligible_news_ids=[frozenset({"A", "B"}), frozenset({"C"})],
        history_news_ids=[frozenset({"A"}), frozenset()],
        availability=availability,
        top_k=1,
        exclude_history=True,
        batch_size=2,
    )

    assert batch.results[0].news_ids == ["B"]
    assert batch.results[1].news_ids == ["C"]
    assert batch.results[0].rejected_history_count >= 1
    assert batch.results[1].rejected_unavailable_count >= 1
    assert batch.batch_count == 1


def test_faiss_load_rejects_incompatible_dimension(tmp_path: Path):
    dense_index = _dense_index()
    loaded = build_faiss_index(dense_index, FaissIndexConfig(index_type="flat"))
    index_path = tmp_path / "flat.index"
    metadata_path = tmp_path / "flat.metadata.json"
    save_faiss_index(loaded, index_path=index_path, metadata_path=metadata_path)

    with pytest.raises(ValueError, match="dimension"):
        load_faiss_index(
            index_path=index_path,
            metadata_path=metadata_path,
            expected_dimension=dense_index.effective_dimension + 1,
        )
