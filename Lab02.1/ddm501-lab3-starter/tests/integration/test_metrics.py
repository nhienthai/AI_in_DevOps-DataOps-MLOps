"""
Integration tests for the Prometheus metrics endpoint.

Monitoring is only worth having if it is correct. An endpoint that returns 200
with the wrong labels is worse than no endpoint - it produces dashboards that
lie. These tests pin the exposition format, the series names, that counters
actually move, and that labels use the route *template* rather than the raw URL.

Run tests:
    pytest tests/integration/test_metrics.py -v
"""

from typing import Dict

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from app import metrics as app_metrics

pytestmark = pytest.mark.integration


def _counter_value(counter: object, **labels: str) -> float:
    """Read the current value of a labelled counter."""
    return float(counter.labels(**labels)._value.get())  # type: ignore[attr-defined]


class TestMetricsEndpoint:
    """GET /metrics - format and contents."""

    def test_metrics_returns_200(self, test_client: TestClient) -> None:
        """The scrape endpoint is reachable."""
        assert test_client.get("/metrics").status_code == 200

    def test_metrics_uses_prometheus_content_type(self, test_client: TestClient) -> None:
        """
        Prometheus dispatches on Content-Type.

        Serving JSON here would make the endpoint unscrapeable while still
        returning 200 - a failure no status-code check would catch.
        """
        response = test_client.get("/metrics")
        assert response.headers["content-type"].startswith(CONTENT_TYPE_LATEST.split(";")[0])

    def test_metrics_body_is_text_exposition_format(self, test_client: TestClient) -> None:
        """The body carries HELP/TYPE lines, as the format requires."""
        body = test_client.get("/metrics").text
        assert "# HELP" in body
        assert "# TYPE" in body

    @pytest.mark.parametrize(
        "series",
        [
            "http_requests_total",
            "http_request_duration_seconds",
            "predictions_total",
            "prediction_cache_events_total",
            "model_loaded",
            "cache_connected",
            "model_info",
        ],
    )
    def test_expected_series_are_exposed(self, test_client: TestClient, series: str) -> None:
        """Every series a dashboard depends on is present."""
        assert series in test_client.get("/metrics").text


class TestMetricsAreRecorded:
    """The middleware and endpoints actually move the counters."""

    def test_request_counter_increments(self, test_client: TestClient) -> None:
        """A handled request is counted, with method, route and status labels."""
        labels = {"method": "GET", "path": "/health", "status": "200"}
        before = _counter_value(app_metrics.REQUEST_COUNT, **labels)
        test_client.get("/health")
        after = _counter_value(app_metrics.REQUEST_COUNT, **labels)
        assert after == before + 1

    def test_prediction_counter_increments(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Each served prediction is counted against its endpoint."""
        before = _counter_value(app_metrics.PREDICTIONS, endpoint="/predict")
        api_client.post("/predict", json=sample_prediction_request)
        after = _counter_value(app_metrics.PREDICTIONS, endpoint="/predict")
        assert after == before + 1

    def test_batch_counter_increments_by_batch_size(
        self, api_client: TestClient, sample_batch_request: Dict[str, object]
    ) -> None:
        """A batch of three counts as three predictions, not one request."""
        before = _counter_value(app_metrics.PREDICTIONS, endpoint="/predict/batch")
        api_client.post("/predict/batch", json=sample_batch_request)
        after = _counter_value(app_metrics.PREDICTIONS, endpoint="/predict/batch")
        assert after == before + len(sample_batch_request["predictions"])  # type: ignore[arg-type]

    def test_cache_events_are_recorded(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Every prediction records a cache hit or miss - hit rate is derivable."""
        before = _counter_value(app_metrics.CACHE_EVENTS, result="miss")
        api_client.post("/predict", json=sample_prediction_request)
        after = _counter_value(app_metrics.CACHE_EVENTS, result="miss")
        assert after >= before

    def test_failed_requests_are_counted_too(self, test_client: TestClient) -> None:
        """
        422s are counted, not only successes.

        An error-rate panel that silently drops errors is the worst kind of
        monitoring bug.
        """
        labels = {"method": "POST", "path": "/predict", "status": "422"}
        before = _counter_value(app_metrics.REQUEST_COUNT, **labels)
        test_client.post("/predict", json={"movie_id": "242"})
        after = _counter_value(app_metrics.REQUEST_COUNT, **labels)
        assert after == before + 1


class TestMetricsLabelling:
    """Label cardinality - the classic way to melt a Prometheus server."""

    def test_labels_use_the_route_template(self, test_client: TestClient) -> None:
        """
        Labels carry "/predict", never "/predict?user=196".

        Labelling on raw URLs would create a new time series per distinct URL,
        which is unbounded cardinality.
        """
        test_client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        body = test_client.get("/metrics").text
        assert 'path="/predict"' in body

    def test_metrics_endpoint_does_not_count_itself(self, test_client: TestClient) -> None:
        """Scrapes must not inflate the request counters they report."""
        test_client.get("/metrics")
        body = test_client.get("/metrics").text
        assert 'path="/metrics"' not in body

    def test_unknown_routes_are_not_labelled(self, test_client: TestClient) -> None:
        """A 404 has no route template, so it creates no series."""
        test_client.get("/definitely-not-a-route")
        assert 'path="/definitely-not-a-route"' not in test_client.get("/metrics").text


class TestMetricsGauges:
    """Gauges describe current state and must be refreshed, not accumulated."""

    def test_model_loaded_gauge_reflects_state(self, test_client: TestClient) -> None:
        """model_loaded agrees with what /health reports."""
        loaded = test_client.get("/health").json()["model_loaded"]
        test_client.get("/metrics")
        assert app_metrics.MODEL_LOADED._value.get() == (1.0 if loaded else 0.0)

    def test_model_loaded_gauge_drops_when_model_is_gone(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Losing the model must be visible in monitoring, not only in /health.

        This is the series an alert would fire on.
        """
        monkeypatch.setattr("app.main.model", None)
        test_client.get("/metrics")
        assert app_metrics.MODEL_LOADED._value.get() == 0.0

    def test_model_info_carries_version_and_type(self, test_client: TestClient) -> None:
        """The deployed version is a label, so dashboards can group by release."""
        body = test_client.get("/metrics").text
        assert 'model_info{type="SVD (Collaborative Filtering)"' in body

    def test_training_metrics_are_published(self, test_client: TestClient) -> None:
        """
        The model's RMSE/MAE sit next to live traffic in Prometheus.

        Skipped when the metadata file is absent, since there is nothing to
        publish in that case.
        """
        info = test_client.get("/model/info").json()
        if "metrics" not in info:
            pytest.skip("models/model_metadata.json not present")
        body = test_client.get("/metrics").text
        assert 'model_training_metric{metric="rmse"}' in body
        assert 'model_training_metric{metric="mae"}' in body


class TestLatencyHeader:
    """The X-Process-Time-Ms header added by the same middleware."""

    def test_latency_header_is_present(self, test_client: TestClient) -> None:
        """Every response carries its own server-side latency."""
        assert "X-Process-Time-Ms" in test_client.get("/health").headers

    def test_latency_header_is_a_positive_number(self, test_client: TestClient) -> None:
        """The value parses as a float and is not negative."""
        value = float(test_client.get("/health").headers["X-Process-Time-Ms"])
        assert value >= 0.0

    def test_latency_header_present_on_errors(self, test_client: TestClient) -> None:
        """Failed requests are timed too - slow failures are still slow."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422
        assert "X-Process-Time-Ms" in response.headers


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
