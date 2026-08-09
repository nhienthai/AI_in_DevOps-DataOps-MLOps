"""Regression tests for the monitored prediction application."""

import pickle
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

import app.main as main_module
import app.model as model_module
from app.middleware import MetricsMiddleware, check_metrics_middleware_ready
from app.model import MovieRatingModel


class FixedEstimator:
    """Small pickleable estimator implementing Surprise's prediction contract."""

    def __init__(self, estimate=3.75):
        self.estimate = estimate

    def predict(self, user_id, movie_id):
        return SimpleNamespace(est=self.estimate)


class RaisingEstimator:
    def __init__(self, error):
        self.error = error

    def predict(self, user_id, movie_id):
        raise self.error


def _sample_value(name, labels=None):
    labels = labels or {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                return sample.value
    return 0


@pytest.fixture
def loaded_model(tmp_path):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(FixedEstimator(), model_file)
    return MovieRatingModel(str(model_path))


@pytest.fixture
def client(monkeypatch, loaded_model):
    monkeypatch.setattr(main_module, "model", loaded_model)
    return TestClient(main_module.app)


def test_info_health_metrics_and_model_endpoints(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["metrics_implemented"] == "10/10"

    health = client.get("/health")
    assert health.json()["status"] == "healthy"
    assert health.json()["model_loaded"] is True

    metrics_info = client.get("/metrics/info")
    assert metrics_info.status_code == 200
    assert metrics_info.json()["metrics_count"] == 10

    model_info = client.get("/model/info")
    assert model_info.status_code == 200
    assert model_info.json()["is_loaded"] is True


def test_single_and_batch_predictions_record_metrics(client):
    single = client.post(
        "/predict", json={"user_id": " 196 ", "movie_id": " 242 "}
    )
    assert single.status_code == 200
    assert single.json()["user_id"] == "196"
    assert single.json()["predicted_rating"] == 3.75

    batch_count_before = _sample_value("ml_batch_prediction_size_count")
    batch = client.post(
        "/predict/batch",
        json={
            "predictions": [
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "186", "movie_id": "302"},
            ]
        },
    )
    assert batch.status_code == 200
    assert batch.json()["total_count"] == 2
    assert batch.json()["avg_latency_ms"] >= 0
    assert _sample_value("ml_batch_prediction_size_count") == (
        batch_count_before + 1
    )


def test_model_metrics_use_dashboard_metric_name(loaded_model):
    output = generate_latest().decode()
    assert "ml_model_info{" in output
    assert "ml_model_info_info" not in output
    assert 'version="1.0.0"' in output


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [(8.2, 5.0), (-3.0, 1.0), (3.456, 3.46)],
)
def test_model_clips_and_rounds_predictions(loaded_model, estimate, expected):
    loaded_model.model = FixedEstimator(estimate)
    assert loaded_model.predict("1", "2") == expected


@pytest.mark.parametrize(
    ("error", "error_type"),
    [(ValueError("bad input"), "validation_error"), (RuntimeError("boom"), "unknown_error")],
)
def test_model_records_prediction_errors(loaded_model, error, error_type):
    loaded_model.model = RaisingEstimator(error)
    before = _sample_value(
        "ml_prediction_errors_total",
        {"error_type": error_type, "model_version": loaded_model.version},
    )

    with pytest.raises(type(error), match=str(error)):
        loaded_model.predict("1", "2")

    after = _sample_value(
        "ml_prediction_errors_total",
        {"error_type": error_type, "model_version": loaded_model.version},
    )
    assert after == before + 1


def test_model_helpers_and_batch(loaded_model):
    ratings = loaded_model.predict_batch([("1", "2"), ("3", "4")])
    assert ratings == [3.75, 3.75]

    rating, latency_ms = loaded_model.predict_with_latency("1", "2")
    assert rating == 3.75
    assert latency_ms >= 0
    assert loaded_model.is_loaded() is True
    assert loaded_model.get_info()["path"] == loaded_model.model_path


def test_model_load_failures_set_unhealthy_metric(tmp_path):
    with pytest.raises(FileNotFoundError):
        MovieRatingModel(str(tmp_path / "missing.pkl"))

    corrupt_path = tmp_path / "corrupt.pkl"
    corrupt_path.write_bytes(b"not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        MovieRatingModel(str(corrupt_path))


def test_model_not_loaded_error(loaded_model):
    loaded_model.model = None
    with pytest.raises(RuntimeError, match="Model not loaded"):
        loaded_model.predict("1", "2")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        loaded_model.predict_batch([("1", "2")])


def test_singleton_lifecycle(monkeypatch):
    created = []

    def factory():
        instance = object()
        created.append(instance)
        return instance

    model_module.reset_model()
    monkeypatch.setattr(model_module, "MovieRatingModel", factory)
    first = model_module.get_model()
    assert model_module.get_model() is first
    assert model_module.reload_model() is not first
    assert len(created) == 2
    model_module.reset_model()


def test_unavailable_model_and_validation_responses(monkeypatch):
    monkeypatch.setattr(main_module, "model", None)
    client = TestClient(main_module.app)

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["status"] == "unhealthy"
    assert client.get("/model/info").json()["is_loaded"] is False
    assert client.post(
        "/predict", json={"user_id": "1", "movie_id": "2"}
    ).status_code == 503
    assert client.post(
        "/predict/batch",
        json={"predictions": [{"user_id": "1", "movie_id": "2"}]},
    ).status_code == 503
    assert client.post(
        "/predict", json={"user_id": " ", "movie_id": "2"}
    ).status_code == 422
    assert client.post(
        "/predict/batch", json={"predictions": []}
    ).status_code == 422


def test_prediction_exceptions_return_500(client, loaded_model):
    loaded_model.model = RaisingEstimator(RuntimeError("prediction failed"))
    response = client.post(
        "/predict", json={"user_id": "1", "movie_id": "2"}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "prediction failed"

    response = client.post(
        "/predict/batch",
        json={"predictions": [{"user_id": "1", "movie_id": "2"}]},
    )
    assert response.status_code == 500


def test_metrics_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(main_module, "METRICS_ENABLED", False)
    response = client.get("/metrics")
    assert response.status_code == 503
    assert response.json()["detail"] == "Metrics disabled"


def test_middleware_records_unhandled_500():
    failing_app = FastAPI()
    failing_app.add_middleware(MetricsMiddleware)

    @failing_app.get("/explode")
    async def explode():
        raise RuntimeError("boom")

    labels = {"method": "GET", "endpoint": "/explode", "status": "500"}
    before = _sample_value("http_requests_total", labels)
    response = TestClient(
        failing_app, raise_server_exceptions=False
    ).get("/explode")
    assert response.status_code == 500
    assert _sample_value("http_requests_total", labels) == before + 1


def test_middleware_readiness():
    assert check_metrics_middleware_ready() == {
        "request_count_ready": True,
        "request_latency_ready": True,
        "middleware_functional": True,
    }
