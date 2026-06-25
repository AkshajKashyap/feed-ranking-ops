.PHONY: install validate-data audit-data prepare-data evaluate-baselines evaluate-baselines-smoke test lint check

install:
	python -m pip install -e ".[dev]"

validate-data:
	python -m feed_ranking_ops.data.validate_layout --data-dir data/raw

audit-data:
	python -m feed_ranking_ops.data.audit_dataset --data-dir data/raw --reports-dir reports

prepare-data:
	python -m feed_ranking_ops.data.prepare_dataset --data-dir data/raw --output-dir data/processed --reports-dir reports

evaluate-baselines:
	python -m feed_ranking_ops.evaluation.run_baselines --processed-dir data/processed --reports-dir reports/baselines

evaluate-baselines-smoke:
	python -m feed_ranking_ops.evaluation.run_baselines --processed-dir data/processed --reports-dir reports/baselines_smoke --limit-impressions 100

test:
	pytest -q

lint:
	ruff check . --no-cache

check: lint test
