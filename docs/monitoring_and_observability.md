# Serving Observability And Offline Monitoring

## Scope

Milestone 7 adds diagnostics around the selected `category_affinity` policy. It does not
change ranking, promotion evidence, or the candidate set. The implementation covers local
request telemetry, in-process latency counters, offline policy slices, and drift-style
distribution comparisons.

This is an auditable local workflow, not a hosted monitoring platform.

## Request Logging

Set the following environment variable before starting the API:

```bash
export FEED_RANKING_OPS_REQUEST_LOG=artifacts/logs/rank_requests.jsonl
uvicorn feed_ranking_ops.serving.app:app --host 127.0.0.1 --port 8000
```

Each schema-valid `POST /rank` call, including service or policy failures, appends one JSON
object containing:

- UTC timestamp and request ID;
- selected policy;
- history, candidate, ranked, missing-candidate, and unknown-history counts;
- empty-history flag;
- request latency in milliseconds;
- warnings, status, and outcome;
- aggregate category counts among the top ten ranked results.

Raw history and candidate IDs are not logged. Clients may supply `X-Request-ID`; otherwise
the service creates a UUID. JSONL writes are optional and failure-tolerant: if no path is
configured, ranking and in-process metrics continue normally. A log write error increments
an internal counter rather than failing the user request.

The files are intended for local development and may still contain operational metadata.
They remain ignored by Git and should be handled according to the environment's privacy and
retention rules.

## API Diagnostics

The API exposes:

- `GET /health`: policy/artifact liveness;
- `GET /policy`: public manifest, evaluation, and limitation metadata;
- `POST /rank`: deterministic logged-candidate ranking;
- `GET /metrics`: process-local counters for rank requests.

`GET /metrics` reports total, successful, and failed requests; average, p50, and p95 latency;
missing-candidate rate; unknown-history rate; empty-history rate; request-logging status; and
log-write errors.

These counters reset when the process restarts and deliberately avoid a Prometheus
dependency.

## Logged Serving Smoke

The logged smoke uses the packaged news catalog and makes three ranking calls:

1. normal known history and candidates;
2. empty history;
3. an unknown candidate.

It also verifies health, policy, and metrics:

```bash
make smoke-serve-logged
```

The unlogged smoke remains available as `make smoke-serve`.

## Offline Monitoring Report

Generate a full report:

```bash
python -m feed_ranking_ops.monitoring.generate_report \
  --processed-dir data/processed \
  --serving-artifacts-dir artifacts/serving \
  --reports-dir reports/monitoring \
  --request-log artifacts/logs/rank_requests.jsonl

make monitor
```

Use a bounded sample for a quick diagnostic:

```bash
make smoke-monitor
```

Outputs:

```text
reports/monitoring/
  monitoring_report.json
  monitoring_report.md
```

The report still succeeds when the request log does not exist and states that request
telemetry was unavailable.

## Offline Policy Slices

The packaged policy is scored directly over processed validation and internal-test
impressions. The scorer is the same implementation used by `POST /rank`.

The report includes:

- overall MRR, NDCG@5/10, Recall@5/10, Hit Rate@5/10, and AUC;
- metrics by history length (`0`, `1-5`, `6-20`, `21+`);
- metrics by candidate count (`1-10`, `11-50`, `51+`);
- empty-history versus non-empty-history metrics;
- metrics grouped by the top-ranked candidate's category;
- known category/subcategory coverage;
- missing-candidate coverage;
- fraction of impressions containing score ties;
- average category concentration among each top ten.

Source candidate position remains the deterministic tie-breaker.

## Drift-Style Checks

Training inputs form the reference profile. Validation and internal test are compared with:

- candidate-category Jensen-Shannon divergence;
- history-length-bucket Jensen-Shannon divergence;
- candidate-count-bucket Jensen-Shannon divergence;
- absolute empty-history-rate difference;
- absolute missing-candidate-rate difference.

When a request log exists, history length, candidate count, empty history, and missing
candidate rates are compared with the training reference. Logged top-result categories are
reported separately because they are not equivalent to the full candidate-category
distribution.

Jensen-Shannon divergence uses base-2 logarithms and ranges from zero for identical
distributions to one for disjoint distributions.

## Limitations

- Diagnostics are offline snapshots and process-local counters, not continuous production
  monitoring.
- The internal test is a chronological holdout from MIND-small train, not an official MIND
  benchmark.
- Slice metrics inherit logged exposure and position bias.
- Small request logs can produce unstable distribution comparisons.
- No alerting, persistence service, dashboard, authentication, or online experimentation is
  included.
