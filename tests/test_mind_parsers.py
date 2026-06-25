from pathlib import Path

import pytest

from feed_ranking_ops.data.parsers import parse_behavior_file, parse_news_file
from feed_ranking_ops.data.schemas import MindParseError


def test_valid_news_parsing_handles_missing_abstract(tmp_path: Path):
    news_path = tmp_path / "news.tsv"
    news_path.write_text(
        "N1\tnews\tlocal\tTitle one\t\thttps://example.test/n1\t[]\t[]\n"
        "N2\tsports\tfootball\tTitle two\tAbstract two\thttps://example.test/n2\t[]\t[]\n",
        encoding="utf-8",
    )

    records, stats = parse_news_file(news_path, "train")

    assert stats.row_count == 2
    assert records[0].news_id == "N1"
    assert records[0].abstract == ""
    assert records[1].title == "Title two"


def test_behavior_parsing_handles_empty_history_labeled_and_unlabeled_candidates(
    tmp_path: Path,
):
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        "1\tU1\t11/15/2019 8:00:00 AM\t\tN1-1 N2-0 N3\n"
        "2\tU1\t2019-11-15T09:00:00+00:00\tN1 N2\tN3\n",
        encoding="utf-8",
    )

    records, stats = parse_behavior_file(behaviors_path, "train")

    assert len(records) == 2
    assert records[0].history == []
    assert records[1].history == ["N1", "N2"]
    assert [candidate.clicked for candidate in records[0].impressions] == [1, 0, None]
    assert stats.empty_history_count == 1
    assert stats.clicked_candidate_count == 1
    assert stats.non_clicked_candidate_count == 1
    assert stats.unlabeled_candidate_count == 2


def test_malformed_impression_tokens_are_counted_in_audit_mode(tmp_path: Path):
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        "1\tU1\t11/15/2019 8:00:00 AM\tN1\tN1-1 BROKEN-x N2-0\n",
        encoding="utf-8",
    )

    records, stats = parse_behavior_file(behaviors_path, "train", strict=False)

    assert len(records) == 1
    assert stats.malformed_impression_token_count == 1
    assert [candidate.news_id for candidate in records[0].impressions] == ["N1", "N2"]

    with pytest.raises(MindParseError, match="malformed impression token"):
        parse_behavior_file(behaviors_path, "train", strict=True)


def test_invalid_timestamps_are_counted_in_audit_mode(tmp_path: Path):
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        "1\tU1\tnot-a-date\tN1\tN1-1\n"
        "2\tU2\t11/15/2019 9:00:00 AM\tN1\tN2-0\n",
        encoding="utf-8",
    )

    records, stats = parse_behavior_file(behaviors_path, "train", strict=False)

    assert len(records) == 1
    assert records[0].impression_id == "2"
    assert stats.invalid_timestamp_count == 1

    with pytest.raises(MindParseError, match="invalid timestamp"):
        parse_behavior_file(behaviors_path, "train", strict=True)


def test_inconsistent_column_count_fails_with_actionable_error(tmp_path: Path):
    news_path = tmp_path / "news.tsv"
    news_path.write_text("N1\ttoo\tfew\n", encoding="utf-8")

    with pytest.raises(MindParseError, match="expected 8 tab-separated columns"):
        parse_news_file(news_path, "train")
