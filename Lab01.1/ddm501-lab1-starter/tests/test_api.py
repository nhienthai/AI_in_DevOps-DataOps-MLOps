"""
Unit tests for Movie Rating Prediction API.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Create test client
client = TestClient(app)


# =============================================================================
# Health Check Tests (PROVIDED)
# =============================================================================
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_format(self):
        """Test that health response has correct format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)

    def test_health_reports_model_loaded(self):
        """A trained model must be present for the rest of the suite to be meaningful."""
        data = client.get("/health").json()

        assert data["model_loaded"] is True, (
            "Model not loaded - run 'python scripts/train_model.py' first"
        )
        assert data["status"] == "healthy"

    def test_health_works_through_lifespan(self):
        """Startup event path (used by uvicorn) also yields a loaded model."""
        with TestClient(app) as lifespan_client:
            data = lifespan_client.get("/health").json()

        assert data["status"] == "healthy"


# =============================================================================
# Root Endpoint Tests (PROVIDED)
# =============================================================================
class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_200(self):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data


# =============================================================================
# Prediction Endpoint Tests
# =============================================================================
class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_response_format(self):
        """Test that prediction response has correct format."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"}
        )
        data = response.json()

        assert "user_id" in data
        assert "movie_id" in data
        assert "predicted_rating" in data
        assert "model_version" in data
        assert isinstance(data["predicted_rating"], float)
        assert isinstance(data["model_version"], str)

    def test_predict_echoes_request_ids(self):
        """Response must echo the IDs that were asked for, not other ones."""
        response = client.post(
            "/predict",
            json={"user_id": "186", "movie_id": "302"}
        )
        data = response.json()

        assert data["user_id"] == "186"
        assert data["movie_id"] == "302"

    def test_predict_is_deterministic(self):
        """The same pair must score identically across calls (no per-request retrain)."""
        payload = {"user_id": "196", "movie_id": "242"}

        first = client.post("/predict", json=payload).json()
        second = client.post("/predict", json=payload).json()

        assert first["predicted_rating"] == second["predicted_rating"]

    def test_predict_missing_user_id(self):
        """Test prediction with missing user_id."""
        response = client.post(
            "/predict",
            json={"movie_id": "242"}  # Missing user_id
        )
        assert response.status_code == 422

    def test_predict_missing_movie_id(self):
        """Test prediction with missing movie_id."""
        response = client.post(
            "/predict",
            json={"user_id": "196"}  # Missing movie_id
        )
        assert response.status_code == 422

    def test_predict_empty_body(self):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_wrong_type_rejected(self):
        """Non-string IDs that cannot be coerced are rejected by validation."""
        response = client.post(
            "/predict",
            json={"user_id": ["196"], "movie_id": {"id": "242"}}
        )
        assert response.status_code == 422

    def test_predict_rejects_get(self):
        """The endpoint is POST-only."""
        response = client.get("/predict")
        assert response.status_code == 405


# =============================================================================
# Edge Case Tests
# =============================================================================
class TestEdgeCases:
    """Edge case tests."""

    def test_predict_unknown_user(self):
        """Unknown user IDs fall back to the global mean instead of failing."""
        response = client.post(
            "/predict",
            json={"user_id": "999999", "movie_id": "242"}
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_unknown_movie(self):
        """Unknown movie IDs fall back to the global mean instead of failing."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "999999"}
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_cold_start_both_unknown(self):
        """Both IDs unknown is still served, not a 500."""
        response = client.post(
            "/predict",
            json={"user_id": "no-such-user", "movie_id": "no-such-movie"}
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_special_characters_in_id(self):
        """IDs with special characters are treated as unknown users, not errors."""
        response = client.post(
            "/predict",
            json={"user_id": "'; DROP TABLE users; --", "movie_id": "<script>alert(1)</script>"}
        )

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_empty_string_id_rejected(self):
        """Empty strings are a client error, not a silent global-mean prediction."""
        response = client.post(
            "/predict",
            json={"user_id": "", "movie_id": "242"}
        )
        assert response.status_code == 422

    def test_predict_whitespace_only_id_rejected(self):
        """Whitespace-only IDs are caught by validation, not by the model wrapper."""
        response = client.post(
            "/predict",
            json={"user_id": "   ", "movie_id": "242"}
        )
        assert response.status_code == 422

    def test_validation_errors_share_one_body_shape(self):
        """Every 422 uses FastAPI's list-of-errors shape, as documented in OpenAPI."""
        bodies = [
            {"movie_id": "242"},               # missing field
            {"user_id": "", "movie_id": "242"},  # empty string
            {"user_id": "   ", "movie_id": "242"},  # whitespace only
            {"user_id": "1" * 500, "movie_id": "242"},  # too long
        ]

        for body in bodies:
            detail = client.post("/predict", json=body).json()["detail"]
            assert isinstance(detail, list), f"detail is not a list for {body}"
            assert "loc" in detail[0] and "msg" in detail[0]

    def test_predict_trims_surrounding_whitespace(self):
        """A padded ID resolves to the same prediction as the bare ID."""
        padded = client.post("/predict", json={"user_id": " 196 ", "movie_id": "242"})
        bare = client.post("/predict", json={"user_id": "196", "movie_id": "242"})

        assert padded.status_code == 200
        assert padded.json() == bare.json()

    def test_predict_overlong_id_rejected(self):
        """Oversized IDs are rejected by the max_length constraint."""
        response = client.post(
            "/predict",
            json={"user_id": "1" * 500, "movie_id": "242"}
        )
        assert response.status_code == 422

    def test_predict_numeric_id_is_coerced_or_rejected(self):
        """Integer IDs must not produce a 500."""
        response = client.post(
            "/predict",
            json={"user_id": 196, "movie_id": 242}
        )
        assert response.status_code in (200, 422)

    def test_predict_malformed_json(self):
        """Malformed JSON bodies produce 422, not a crash."""
        response = client.post(
            "/predict",
            content="{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_rating_is_rounded_to_two_decimals(self):
        """Predictions are rounded for a stable API contract."""
        rating = client.post(
            "/predict", json={"user_id": "196", "movie_id": "242"}
        ).json()["predicted_rating"]

        assert round(rating, 2) == rating


# =============================================================================
# Model Info Endpoint Tests
# =============================================================================
class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_version(self):
        """Test that model info contains version."""
        data = client.get("/model/info").json()

        assert "model_version" in data
        assert isinstance(data["model_version"], str)
        assert data["model_version"]

    def test_model_info_reports_type_and_load_state(self):
        """Model info exposes the algorithm and whether it is serving."""
        data = client.get("/model/info").json()

        assert "SVD" in data["model_type"]
        assert data["is_loaded"] is True

    def test_model_info_exposes_training_metrics(self):
        """Training metrics from train_model.py are surfaced for monitoring."""
        data = client.get("/model/info").json()

        assert "metrics" in data, "Run 'python scripts/train_model.py' to write metadata"
        assert data["metrics"]["rmse"] > 0
        assert data["metrics"]["mae"] > 0


# =============================================================================
# Batch Prediction Tests (BONUS)
# =============================================================================
class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint (BONUS)."""

    def test_batch_predict_multiple_items(self):
        """Test batch prediction with multiple items."""
        response = client.post(
            "/predict/batch",
            json={
                "predictions": [
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "186", "movie_id": "302"},
                    {"user_id": "22", "movie_id": "377"},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3
        assert all(1.0 <= p["predicted_rating"] <= 5.0 for p in data["predictions"])

    def test_batch_predict_preserves_order(self):
        """Results come back aligned with the request order."""
        pairs = [
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "186", "movie_id": "302"},
        ]

        data = client.post("/predict/batch", json={"predictions": pairs}).json()

        assert [(p["user_id"], p["movie_id"]) for p in data["predictions"]] == [
            (p["user_id"], p["movie_id"]) for p in pairs
        ]

    def test_batch_matches_single_predictions(self):
        """Batch and single endpoints must agree on the same pair."""
        pair = {"user_id": "196", "movie_id": "242"}

        single = client.post("/predict", json=pair).json()["predicted_rating"]
        batch = client.post(
            "/predict/batch", json={"predictions": [pair]}
        ).json()["predictions"][0]["predicted_rating"]

        assert single == batch

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"predictions": []})

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["predictions"] == []

    def test_batch_predict_missing_field(self):
        """A malformed item invalidates the whole request."""
        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196"}]},  # missing movie_id
        )
        assert response.status_code == 422

    def test_batch_predict_over_limit_rejected(self):
        """Batches larger than 100 items are rejected to bound latency."""
        response = client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196", "movie_id": "242"}] * 101},
        )
        assert response.status_code == 422


# =============================================================================
# Model wrapper unit tests
# =============================================================================
class TestModelWrapper:
    """Direct tests of the MovieRatingModel wrapper."""

    def test_missing_model_file_raises(self):
        """A missing .pkl surfaces as FileNotFoundError, not a silent None model."""
        from app.model import MovieRatingModel

        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="models/does_not_exist.pkl")

    def test_predict_batch_matches_predict(self):
        """predict_batch is consistent with repeated predict calls."""
        from app.main import model

        pairs = [("196", "242"), ("186", "302")]

        assert model.predict_batch(pairs) == [model.predict(u, m) for u, m in pairs]

    def test_known_and_unknown_ids(self):
        """The wrapper can distinguish trained IDs from cold-start ones."""
        from app.main import model

        assert model.is_known_user("196")
        assert model.is_known_movie("242")
        assert not model.is_known_user("999999")
        assert not model.is_known_movie("999999")

    def test_predict_rejects_blank_ids(self):
        """Blank IDs raise ValueError at the wrapper level."""
        from app.main import model

        with pytest.raises(ValueError):
            model.predict("   ", "242")


# =============================================================================
# Prometheus metrics tests (BONUS)
# =============================================================================
class TestMetricsEndpoint:
    """Tests for the /metrics endpoint scraped by Prometheus."""

    def test_metrics_returns_prometheus_text_format(self):
        """Prometheus needs the text exposition format, not JSON."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    def test_metrics_exposes_expected_series(self):
        """The metrics the dashboards depend on are all present."""
        client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        body = client.get("/metrics").text

        for name in (
            "http_requests_total",
            "http_request_duration_seconds",
            "predictions_total",
            "prediction_cache_events_total",
            "model_loaded",
            "model_info",
            "model_training_metric",
        ):
            assert name in body, f"{name} missing from /metrics"

    def test_request_counter_increments(self):
        """A served request moves the counter for its route."""
        def counter_value() -> float:
            for line in client.get("/metrics").text.splitlines():
                if line.startswith('http_requests_total{') and 'path="/predict"' in line:
                    return float(line.rsplit(" ", 1)[1])
            return 0.0

        before = counter_value()
        client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        assert counter_value() == before + 1

    def test_metrics_labels_use_route_template(self):
        """Labels use the route template so cardinality stays bounded."""
        body = client.get("/metrics").text

        assert 'path="/predict"' in body
        # The scrape endpoint itself is excluded to avoid self-referential noise.
        assert 'path="/metrics"' not in body

    def test_model_loaded_gauge_reflects_state(self):
        """model_loaded is 1 while a model is serving."""
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("model_loaded "):
                assert float(line.split()[1]) == 1.0
                return
        pytest.fail("model_loaded gauge not found")


# =============================================================================
# Redis cache tests (BONUS)
# =============================================================================
class TestPredictionCache:
    """The cache is an optimisation: it must never break a prediction."""

    def test_cache_disabled_by_default(self):
        """A plain `uvicorn app.main:app` run needs no Redis."""
        from app.cache import PredictionCache

        cache = PredictionCache(enabled=False)

        assert cache.get("196", "242") is None
        assert cache.is_connected() is False
        cache.set("196", "242", 3.8)  # no-op, must not raise

    def test_unreachable_redis_does_not_raise(self):
        """An enabled-but-unreachable cache degrades instead of failing."""
        from app.cache import PredictionCache

        cache = PredictionCache(
            enabled=True, url="redis://127.0.0.1:6399/0", timeout=0.05
        )

        assert cache.get("196", "242") is None
        assert cache.is_connected() is False
        cache.set("196", "242", 3.8)  # must not raise

    def test_api_serves_predictions_without_redis(self):
        """The test suite itself runs with no Redis, and /predict still works."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})

        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_cache_key_includes_model_version(self):
        """Retraining under a new version must not serve stale ratings."""
        from app.cache import PredictionCache
        from app.config import MODEL_VERSION

        key = PredictionCache._key("196", "242")

        assert MODEL_VERSION in key
        assert key.endswith(":196:242")

    def test_model_info_reports_cache_state(self):
        """/model/info tells operators whether caching is on."""
        data = client.get("/model/info").json()

        assert data["cache_enabled"] is False


# =============================================================================
# Documentation tests
# =============================================================================
class TestDocumentation:
    """Swagger / OpenAPI documentation is part of the deliverable."""

    def test_swagger_ui_available(self):
        """/docs serves the Swagger UI."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_documents_endpoints(self):
        """All public endpoints appear in the OpenAPI schema."""
        schema = client.get("/openapi.json").json()

        for path in ("/health", "/predict", "/predict/batch", "/model/info"):
            assert path in schema["paths"], f"{path} missing from OpenAPI schema"

    def test_every_endpoint_declares_a_response_schema(self):
        """No JSON endpoint documents a bare `{}` 200 body."""
        schema = client.get("/openapi.json").json()

        for path, methods in schema["paths"].items():
            for method, spec in methods.items():
                content = spec["responses"]["200"]["content"]
                assert content, f"{method.upper()} {path} declares no 200 content"
                # /metrics serves the Prometheus text format, not JSON.
                json_body = content.get("application/json")
                if json_body is not None:
                    assert json_body["schema"], f"{method.upper()} {path} has an empty 200 schema"

    def test_documented_422_matches_actual_422(self):
        """The documented validation-error schema is the one the API really returns."""
        schema = client.get("/openapi.json").json()
        ref = schema["paths"]["/predict"]["post"]["responses"]["422"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        documented = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]

        # Documented `detail` is an array...
        assert documented["properties"]["detail"]["type"] == "array"
        # ...and so is the real one.
        actual = client.post("/predict", json={"movie_id": "242"}).json()
        assert isinstance(actual["detail"], list)

    def test_endpoints_have_descriptions(self):
        """Every operation carries a docstring-derived description for Swagger."""
        schema = client.get("/openapi.json").json()

        for path, methods in schema["paths"].items():
            for method, spec in methods.items():
                assert spec.get("description"), f"{method.upper()} {path} has no description"


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
