#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${FEED_RANKING_OPS_ROOT:-$SCRIPT_ROOT}"
if [[ -z "${PYTHON_BIN:-}" && -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

cd "$PROJECT_ROOT"

required_processed=(
  "data/processed/news.parquet"
  "data/processed/train_behaviors.parquet"
  "data/processed/validation_behaviors.parquet"
  "data/processed/test_behaviors.parquet"
  "data/processed/split_metadata.json"
)

for path in "${required_processed[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Demo requires prepared MIND data; missing: $path" >&2
    echo "Prepare it first with: make prepare-data-train-only" >&2
    exit 1
  fi
done

for path in \
  "reports/baselines/validation_metrics.json" \
  "reports/baselines/test_metrics.json" \
  "reports/ltr/validation_metrics.json" \
  "reports/ltr/test_metrics.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Demo requires existing baseline and LTR reports; missing: $path" >&2
    echo "Run: make evaluate-baselines && make evaluate-ltr" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m feed_ranking_ops.ranking.select_policy \
  --baseline-reports-dir reports/baselines \
  --ltr-reports-dir reports/ltr \
  --processed-dir data/processed \
  --reports-dir reports/model_selection \
  --artifacts-dir artifacts/serving

"$PYTHON_BIN" -m feed_ranking_ops.serving.smoke \
  --manifest artifacts/serving/policy_manifest.json

"$PYTHON_BIN" -m feed_ranking_ops.portfolio.generate_report \
  --reports-dir reports \
  --artifacts-dir artifacts \
  --output-dir reports/portfolio

echo "Demo completed."
echo "Policy report: reports/model_selection/promotion_report.md"
echo "Serving manifest: artifacts/serving/policy_manifest.json"
echo "Portfolio summary: reports/portfolio/portfolio_summary.md"
