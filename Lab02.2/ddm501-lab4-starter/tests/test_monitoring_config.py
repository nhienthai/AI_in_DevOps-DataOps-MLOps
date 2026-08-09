"""Regression tests for Prometheus and Grafana wiring."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _compact(path):
    return " ".join(path.read_text(encoding="utf-8").split())


def test_api_error_alert_aggregates_status_series():
    alerts = _compact(ROOT / "prometheus" / "alerts" / "api_alerts.yml")
    assert 'sum(rate(http_requests_total{status=~"5.."}[5m]))' in alerts
    assert "clamp_min(sum(rate(http_requests_total[5m]))" in alerts


def test_ml_error_alert_aligns_labels_and_counts_all_attempts():
    alerts = _compact(ROOT / "prometheus" / "alerts" / "ml_alerts.yml")
    assert (
        "sum by (model_version) (rate(ml_prediction_errors_total[5m]))"
        in alerts
    )
    assert "sum by (model_version) (rate(ml_predictions_total[5m]))" in alerts


def test_grafana_queries_and_datasource_uid_match_metrics():
    dashboard = json.loads(
        (ROOT / "grafana" / "dashboards" / "ml_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]
    assert "ml_model_info" in expressions
    assert any(
        "sum(rate(ml_prediction_errors_total[5m]))" in expr
        for expr in expressions
    )
    assert any("ml_prediction_value_sum" in expr for expr in expressions)

    datasource = _compact(
        ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    )
    assert "uid: prometheus" in datasource


def test_runtime_image_builds_surprise_without_shipping_compiler():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "FROM python:3.10-slim AS builder" in dockerfile
    assert "build-essential" in dockerfile
    assert "COPY --from=builder /wheels /wheels" in dockerfile
    assert "scikit-surprise==1.1.4" in requirements
