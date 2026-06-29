.PHONY: install validate-data validate-data-train-only audit-data audit-data-train-only prepare-data prepare-data-train-only evaluate-baselines evaluate-baselines-smoke evaluate-retrieval evaluate-retrieval-smoke evaluate-ann evaluate-ann-smoke evaluate-ltr evaluate-ltr-smoke select-policy serve smoke-serve test test-ann lint check

install:
	python -m pip install -e ".[dev]"

validate-data:
	python -m feed_ranking_ops.data.validate_layout --data-dir data/raw

validate-data-train-only:
	python -m feed_ranking_ops.data.validate_layout --data-dir data/raw --protocol train_only_chronological

audit-data:
	python -m feed_ranking_ops.data.audit_dataset --data-dir data/raw --reports-dir reports

audit-data-train-only:
	python -m feed_ranking_ops.data.audit_dataset --data-dir data/raw --reports-dir reports --protocol train_only_chronological

prepare-data:
	python -m feed_ranking_ops.data.prepare_dataset --data-dir data/raw --output-dir data/processed --reports-dir reports

prepare-data-train-only:
	python -m feed_ranking_ops.data.prepare_dataset --data-dir data/raw --output-dir data/processed --reports-dir reports --protocol train_only_chronological

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
	@set -eu; \
	tmp_dir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	python -c "import faiss" 2>/dev/null || { \
		echo "ANN smoke test requires FAISS. Install it with: python -m pip install -e \".[ann]\"" >&2; \
		exit 1; \
	}; \
	python -m feed_ranking_ops.data.prepare_dataset \
		--data-dir tests/fixtures/mindsmall_demo \
		--output-dir "$$tmp_dir/processed" \
		--reports-dir "$$tmp_dir/prepare_reports"; \
	python -m feed_ranking_ops.retrieval.run_ann_benchmark \
		--processed-dir "$$tmp_dir/processed" \
		--reports-dir "$$tmp_dir/ann_reports" \
		--limit-queries 2 \
		--svd-dims 2 \
		--hnsw-m 4 \
		--ef-construction 8 \
		--ef-search 8 \
		--oversampling 2 \
		--top-k 10 \
		--seed 42 \
		--faiss-threads 1; \
	echo "Synthetic ANN smoke test passed; temporary metrics are not benchmark results."

evaluate-ltr:
	python -m feed_ranking_ops.ranking.run_ltr --processed-dir data/processed --reports-dir reports/ltr

evaluate-ltr-smoke:
	python -m feed_ranking_ops.ranking.run_ltr --processed-dir data/processed --reports-dir reports/ltr_smoke --limit-impressions 1000

select-policy:
	python -m feed_ranking_ops.ranking.select_policy --baseline-reports-dir reports/baselines --ltr-reports-dir reports/ltr --processed-dir data/processed --reports-dir reports/model_selection --artifacts-dir artifacts/serving

serve:
	uvicorn feed_ranking_ops.serving.app:app --host 127.0.0.1 --port 8000

smoke-serve:
	python -m feed_ranking_ops.serving.smoke --manifest artifacts/serving/policy_manifest.json

test:
	pytest -q

test-ann:
	pytest -q tests/test_ann_retrieval.py tests/test_faiss_backend.py

lint:
	ruff check . --no-cache

check: lint test
