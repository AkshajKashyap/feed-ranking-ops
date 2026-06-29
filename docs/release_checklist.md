# Release Checklist

## Quality

- [ ] Run `ruff check . --no-cache`.
- [ ] Run `pytest -q`.
- [ ] Run `make check`.
- [ ] Run `git diff --check`.
- [ ] Confirm tests do not require external downloads or real MIND data.

## Data And Artifacts

- [ ] Confirm `data/raw/`, `data/processed/`, `artifacts/`, and generated reports are ignored.
- [ ] Run `git status --short` and verify no MIND TSV or Parquet data is staged.
- [ ] Confirm request logs remain ignored and contain no raw history/candidate IDs.
- [ ] Regenerate `reports/portfolio/portfolio_summary.{json,md}`.

## Demo And Serving

- [ ] Run `bash scripts/generate_demo.sh`.
- [ ] Run `make smoke-serve`.
- [ ] Run `make smoke-serve-logged` when local log verification is appropriate.
- [ ] Run `make smoke-monitor`.
- [ ] Verify `/health`, `/policy`, `/rank`, and `/metrics`.
- [ ] Confirm the selected policy is `category_affinity`.

## Documentation

- [ ] Review `README.md` results and internal-holdout warning.
- [ ] Review `docs/architecture.md`.
- [ ] Review `docs/model_card.md`.
- [ ] Review `docs/experimental_methodology.md`.
- [ ] Review monitoring/privacy limitations.
- [ ] Confirm no result is described as an official MIND validation score.

## Release Metadata

- [ ] Run `feed-ranking-ops --version`.
- [ ] Run `make project-info`.
- [ ] Run `make release-check`.
- [ ] Confirm package version and portfolio version agree.
- [ ] Confirm Git working tree is clean.
- [ ] Push the release commit.
- [ ] Optionally create an annotated Git tag after review.
