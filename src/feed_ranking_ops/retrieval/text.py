from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from feed_ranking_ops.evaluation.processed import BehaviorImpression, NewsItem

TextConfigName = Literal["title", "title_abstract", "title_abstract_category"]


@dataclass(frozen=True)
class TextConfig:
    name: TextConfigName


@dataclass
class ArticleTextIndex:
    text_config: TextConfig
    vectorizer: TfidfVectorizer | None
    article_ids: list[str]
    article_to_row: dict[str, int]
    article_matrix: sparse.csr_matrix
    fitting_article_ids: list[str]

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self.vectorizer is not None else 0


def fit_article_text_index(
    *,
    news: dict[str, NewsItem],
    fitting_behaviors: list[BehaviorImpression],
    text_config: TextConfig,
) -> ArticleTextIndex:
    article_ids = sorted(news)
    fitting_article_ids = sorted(
        {
            news_id
            for behavior in fitting_behaviors
            for news_id in [
                *behavior.history_news_ids,
                *(candidate.news_id for candidate in behavior.candidates),
            ]
            if news_id in news
        }
    )
    fitting_texts = [
        text
        for news_id in fitting_article_ids
        if (text := build_article_text(news[news_id], text_config)).strip()
    ]
    if not fitting_texts:
        matrix = sparse.csr_matrix((len(article_ids), 0), dtype=float)
        return ArticleTextIndex(
            text_config=text_config,
            vectorizer=None,
            article_ids=article_ids,
            article_to_row={news_id: index for index, news_id in enumerate(article_ids)},
            article_matrix=matrix,
            fitting_article_ids=fitting_article_ids,
        )
    vectorizer = TfidfVectorizer()
    vectorizer.fit(fitting_texts)
    matrix = vectorizer.transform(
        [build_article_text(news[news_id], text_config) for news_id in article_ids]
    )
    return ArticleTextIndex(
        text_config=text_config,
        vectorizer=vectorizer,
        article_ids=article_ids,
        article_to_row={news_id: index for index, news_id in enumerate(article_ids)},
        article_matrix=matrix.tocsr(),
        fitting_article_ids=fitting_article_ids,
    )


def build_article_text(item: NewsItem, config: TextConfig) -> str:
    if config.name == "title":
        parts = [item.title]
    elif config.name == "title_abstract":
        parts = [item.title, item.abstract]
    elif config.name == "title_abstract_category":
        parts = [
            item.title,
            item.abstract,
            f"category_{item.category}",
            f"subcategory_{item.subcategory}",
        ]
    else:
        raise ValueError(f"Unknown text configuration: {config.name}")
    return " ".join(part.strip() for part in parts if part and part.strip())


def sparse_memory_bytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
