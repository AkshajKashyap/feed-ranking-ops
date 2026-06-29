from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

from feed_ranking_ops.serving.schemas import RankRequestLogEvent


class ServingObservability:
    def __init__(
        self,
        *,
        request_log_path: Path | None,
        latency_window_size: int = 10_000,
    ) -> None:
        if latency_window_size <= 0:
            raise ValueError("latency_window_size must be positive")
        self.request_log_path = request_log_path
        self._latencies: deque[float] = deque(maxlen=latency_window_size)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_candidate_ids = 0
        self._total_missing_candidates = 0
        self._total_history_ids = 0
        self._total_unknown_history_ids = 0
        self._empty_history_requests = 0
        self._request_log_write_errors = 0

    @property
    def logging_enabled(self) -> bool:
        return self.request_log_path is not None

    def record(self, event: RankRequestLogEvent) -> None:
        with self._lock:
            self._total_requests += 1
            if event.status == "success":
                self._successful_requests += 1
            else:
                self._failed_requests += 1
            self._latencies.append(event.latency_ms)
            self._total_candidate_ids += event.candidate_id_count
            self._total_missing_candidates += event.missing_candidate_count
            self._total_history_ids += event.history_id_count
            self._total_unknown_history_ids += event.unknown_history_count
            self._empty_history_requests += int(event.empty_history)
            if self.request_log_path is not None:
                self._append_event(event)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            total = self._total_requests
            return {
                "total_requests": total,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "average_latency_ms": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "missing_candidate_rate": _rate(
                    self._total_missing_candidates,
                    self._total_candidate_ids,
                ),
                "unknown_history_rate": _rate(
                    self._total_unknown_history_ids,
                    self._total_history_ids,
                ),
                "empty_history_rate": _rate(
                    self._empty_history_requests,
                    total,
                ),
                "request_logging_enabled": self.logging_enabled,
                "request_log_write_errors": self._request_log_write_errors,
            }

    def _append_event(self, event: RankRequestLogEvent) -> None:
        assert self.request_log_path is not None
        try:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        event.model_dump(mode="json"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        except OSError:
            self._request_log_write_errors += 1


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
