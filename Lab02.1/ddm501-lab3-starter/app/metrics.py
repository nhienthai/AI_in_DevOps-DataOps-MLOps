"""
Prometheus metrics for the Movie Rating API.

Exposed at GET /metrics in the text exposition format that Prometheus scrapes.
Carried over from Lab 1, where the Grafana/Prometheus stack was built.
"""

from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# HTTP-level metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests handled.",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    # Prediction requests are sub-millisecond; the default buckets start too coarse.
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# =============================================================================
# Domain metrics
# =============================================================================

PREDICTIONS = Counter(
    "predictions_total",
    "Ratings predicted, by endpoint.",
    ["endpoint"],
)

CACHE_EVENTS = Counter(
    "prediction_cache_events_total",
    "Prediction cache lookups by outcome.",
    ["result"],  # hit | miss
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 when the model is loaded and serving, 0 otherwise.",
)

CACHE_CONNECTED = Gauge(
    "cache_connected",
    "1 when the Redis cache is reachable, 0 otherwise.",
)

MODEL_INFO = Gauge(
    "model_info",
    "Model metadata as labels; the value is always 1.",
    ["version", "type"],
)

# Cross-validation scores of the deployed model, so drift is visible in Grafana
# next to the live traffic.
MODEL_TRAINING_METRIC = Gauge(
    "model_training_metric",
    "Cross-validation score recorded at training time.",
    ["metric"],  # rmse | mae
)


def record_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    """Record one handled HTTP request."""
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_seconds)


def record_cache(hit: bool) -> None:
    """Record a cache lookup outcome."""
    CACHE_EVENTS.labels(result="hit" if hit else "miss").inc()


def record_predictions(endpoint: str, count: int = 1) -> None:
    """Record predictions served by an endpoint."""
    PREDICTIONS.labels(endpoint=endpoint).inc(count)


def set_model_state(
    version: str,
    model_type: str,
    loaded: bool,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish model identity, load state, and training scores."""
    MODEL_LOADED.set(1 if loaded else 0)
    MODEL_INFO.labels(version=version, type=model_type).set(1)
    for name in ("rmse", "mae"):
        value = (metrics or {}).get(name)
        if value is not None:
            MODEL_TRAINING_METRIC.labels(metric=name).set(float(value))
