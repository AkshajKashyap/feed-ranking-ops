.PHONY: install validate-data audit-data prepare-data evaluate-baselines evaluate-baselines-smoke evaluate-retrieval evaluate-retrieval-smoke evaluate-ann evaluate-ann-smoke test test-ann lint check

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

evaluate-retrieval:
	python -m feed_ranking_ops.retrieval.run_exact_retrieval --processed-dir data/processed --reports-dir reports/retrieval

evaluate-retrieval-smoke:
	python -m feed_ranking_ops.retrieval.run_exact_retrieval --processed-dir data/processed --reports-dir reports/retrieval_smoke --limit-queries 100

evaluate-ann:
	python -m feed_ranking_ops.retrieval.run_ann_benchmark --processed-dir data/processed --reports-dir reports/ann

evaluate-ann-smoke:
	python -m feed_ranking_ops.retrieval.run_ann_benchmark --processed-dir data/processed --reports-dir reports/ann_smoke --limit-queries 100 --svd-dims 32,64 --ef-search 64 --oversampling 4

test:
	pytest -q

test-ann:
	pytest -q tests/test_ann_retrieval.py tests/test_faiss_backend.py

lint:
	ruff check . --no-cache

check: lint test
