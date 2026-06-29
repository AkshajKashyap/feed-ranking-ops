#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
fi

make check
python -m feed_ranking_ops.cli --version
python -m feed_ranking_ops.cli project-info

required_docs=(
  "README.md"
  "docs/architecture.md"
  "docs/model_card.md"
  "docs/experimental_methodology.md"
  "docs/release_checklist.md"
)
for path in "${required_docs[@]}"; do
  test -f "$path"
done

bash scripts/generate_demo.sh
make smoke-monitor
git diff --check

echo "Release checks passed."
