.PHONY: install validate-data audit-data prepare-data test lint check

install:
	python -m pip install -e ".[dev]"

validate-data:
	python -m feed_ranking_ops.data.validate_layout --data-dir data/raw

audit-data:
	python -m feed_ranking_ops.data.audit_dataset --data-dir data/raw --reports-dir reports

prepare-data:
	python -m feed_ranking_ops.data.prepare_dataset --data-dir data/raw --output-dir data/processed --reports-dir reports

test:
	pytest -q

lint:
	ruff check .

check: lint test
