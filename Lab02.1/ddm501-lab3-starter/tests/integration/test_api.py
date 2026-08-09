"""
Integration tests for API endpoints.

Level 2 of the ML testing pyramid: the FastAPI app, its schemas and the model
wrapper exercised together through real HTTP calls (Starlette TestClient).

Run tests:
    pytest tests/integration/test_api.py -v
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.model import ModelNotLoadedError

pytestmark = pytest.mark.integration


class TestHealthEndpoint:
    """Integration tests for /health endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_health_returns_200(self, test_client: TestClient) -> None:
        """Test that health endpoint returns 200 status code."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client: TestClient) -> None:
        """Test that health response has status field."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data

    # =========================================================================
    # TODO 1: Additional Health Tests
    # =========================================================================

    def test_health_response_has_model_loaded_field(self, test_client: TestClient) -> None:
        """The health payload reports whether the model is usable."""
        data = test_client.get("/health").json()
        assert "model_loaded" in data

    def test_health_model_loaded_is_boolean(self, test_client: TestClient) -> None:
        """model_loaded is a real boolean, so orchestrators can act on it."""
        data = test_client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)

    def test_health_status_matches_model_loaded(self, test_client: TestClient) -> None:
        """status and model_loaded must never disagree."""
        data = test_client.get("/health").json()
        assert data["status"] == ("healthy" if data["model_loaded"] else "unhealthy")

    def test_health_reports_unhealthy_without_model(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the model gone, /health still answers - but says unhealthy."""
        monkeypatch.setattr("app.main.model", None)
        data = test_client.get("/health").json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestRootEndpoint:
    """Integration tests for / endpoint."""

    def test_root_returns_200(self, test_client: TestClient) -> None:
        """Test that root endpoint returns 200 status code."""
        response = test_client.get("/")
        assert response.status_code == 200

    # =========================================================================
    # TODO 2: Root Endpoint Tests
    # =========================================================================

    def test_root_contains_api_info(self, test_client: TestClient) -> None:
        """The landing payload advertises name, version and where the docs are."""
        data = test_client.get("/").json()
        for field in ("name", "version", "description", "docs", "health"):
            assert field in data, f"missing '{field}' in root response"

    def test_root_links_are_reachable(self, test_client: TestClient) -> None:
        """The health link advertised at / actually resolves."""
        data = test_client.get("/").json()
        assert test_client.get(data["health"]).status_code == 200

    def test_openapi_schema_is_served(self, test_client: TestClient) -> None:
        """The OpenAPI contract is published for consumers."""
        schema = test_client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]
        assert "/predict/batch" in schema["paths"]


class TestPredictEndpoint:
    """Integration tests for /predict endpoint."""

    # =========================================================================
    # Provided Tests
    # =========================================================================

    def test_predict_valid_request_returns_200(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Test that valid prediction request returns 200."""
        response = api_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    # =========================================================================
    # TODO 3: Response Structure Tests
    # =========================================================================

    def test_predict_response_has_predicted_rating(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """The response carries the prediction itself."""
        data = api_client.post("/predict", json=sample_prediction_request).json()
        assert "predicted_rating" in data
        assert isinstance(data["predicted_rating"], float)

    def test_predict_response_has_user_id(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """The response echoes the user_id it was asked about."""
        data = api_client.post("/predict", json=sample_prediction_request).json()
        assert data["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """The response echoes the movie_id it was asked about."""
        data = api_client.post("/predict", json=sample_prediction_request).json()
        assert data["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_has_model_version(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Every prediction is traceable to a model version."""
        data = api_client.post("/predict", json=sample_prediction_request).json()
        assert data["model_version"]

    def test_predict_response_rating_in_valid_range(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """The served rating stays on the 1-5 scale."""
        data = api_client.post("/predict", json=sample_prediction_request).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_trims_whitespace_in_ids(self, api_client: TestClient) -> None:
        """IDs are normalised by the schema before reaching the model."""
        data = api_client.post("/predict", json={"user_id": " 196 ", "movie_id": " 242 "}).json()
        assert data["user_id"] == "196"
        assert data["movie_id"] == "242"

    # =========================================================================
    # TODO 4: Validation Error Tests
    # =========================================================================

    def test_predict_missing_user_id_returns_422(self, test_client: TestClient) -> None:
        """A request without user_id is rejected before touching the model."""
        response = test_client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client: TestClient) -> None:
        """A request without movie_id is rejected before touching the model."""
        response = test_client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, test_client: TestClient) -> None:
        """An empty JSON object is not a valid request."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client: TestClient) -> None:
        """Malformed JSON is reported as a validation error, not a 500."""
        response = test_client.post(
            "/predict",
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_all_invalid_requests_return_422(
        self, test_client: TestClient, invalid_prediction_requests: List[Dict[str, str]]
    ) -> None:
        """Every catalogued bad request shape is rejected with 422."""
        for payload in invalid_prediction_requests:
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 422, f"{payload} was accepted"

    def test_predict_validation_error_body_is_informative(self, test_client: TestClient) -> None:
        """422 responses name the offending field so clients can fix the call."""
        body = test_client.post("/predict", json={"movie_id": "242"}).json()
        assert "detail" in body
        assert any("user_id" in str(item.get("loc", "")) for item in body["detail"])

    # =========================================================================
    # TODO 5: Multiple Request Tests
    # =========================================================================

    def test_predict_multiple_valid_requests(
        self, api_client: TestClient, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Every known pair is servable and in range."""
        for pair in known_user_movie_pairs:
            response = api_client.post(
                "/predict", json={"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            )
            assert response.status_code == 200, f"{pair} failed"
            assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_is_idempotent_across_requests(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """Repeating the same call over HTTP returns the same rating."""
        first = api_client.post("/predict", json=sample_prediction_request).json()
        second = api_client.post("/predict", json=sample_prediction_request).json()
        assert first["predicted_rating"] == second["predicted_rating"]

    def test_predict_unknown_ids_still_return_valid_rating(self, api_client: TestClient) -> None:
        """Cold-start requests degrade to a default rating rather than erroring."""
        response = api_client.post("/predict", json={"user_id": "99999", "movie_id": "99999"})
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0


class TestBatchPredictEndpoint:
    """Integration tests for /predict/batch endpoint."""

    # =========================================================================
    # TODO 6: Batch Prediction Tests
    # =========================================================================

    def test_batch_predict_returns_200(
        self, api_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """A well-formed batch is accepted."""
        response = api_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(
        self, api_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """total_count matches both the request size and the payload length."""
        data = api_client.post("/predict/batch", json=sample_batch_request).json()
        expected = len(sample_batch_request["predictions"])
        assert data["total_count"] == expected
        assert len(data["predictions"]) == expected

    def test_batch_predict_all_ratings_in_range(
        self, api_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """No item in the batch escapes the rating scale."""
        data = api_client.post("/predict/batch", json=sample_batch_request).json()
        for item in data["predictions"]:
            assert 1.0 <= item["predicted_rating"] <= 5.0

    def test_batch_predict_preserves_order(
        self, api_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Result i answers request i - clients rely on positional matching."""
        data = api_client.post("/predict/batch", json=sample_batch_request).json()
        for sent, got in zip(sample_batch_request["predictions"], data["predictions"]):
            assert (got["user_id"], got["movie_id"]) == (sent["user_id"], sent["movie_id"])

    def test_batch_matches_single_predictions(
        self, api_client: TestClient, sample_batch_request: Dict[str, Any]
    ) -> None:
        """Batching is an optimisation, not a different model path."""
        batch = api_client.post("/predict/batch", json=sample_batch_request).json()
        for sent, got in zip(sample_batch_request["predictions"], batch["predictions"]):
            single = api_client.post("/predict", json=sent).json()
            assert single["predicted_rating"] == got["predicted_rating"]

    def test_batch_predict_empty_list_returns_422(self, test_client: TestClient) -> None:
        """An empty batch is a client error."""
        response = test_client.post("/predict/batch", json={"predictions": []})
        assert response.status_code == 422

    def test_batch_predict_over_limit_returns_422(self, test_client: TestClient) -> None:
        """Batches larger than 100 items are refused, protecting the service."""
        payload = {"predictions": [{"user_id": str(i), "movie_id": "1"} for i in range(101)]}
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 422


class TestErrorHandling:
    """Tests for API error handling."""

    # =========================================================================
    # TODO 7: Error Handling Tests
    # =========================================================================

    def test_404_for_unknown_endpoint(self, test_client: TestClient) -> None:
        """Unknown routes return 404."""
        assert test_client.get("/unknown").status_code == 404

    def test_method_not_allowed_get_predict(self, test_client: TestClient) -> None:
        """GET /predict is not allowed - the endpoint is POST-only."""
        assert test_client.get("/predict").status_code == 405

    def test_method_not_allowed_post_health(self, test_client: TestClient) -> None:
        """POST /health is not allowed - the endpoint is GET-only."""
        assert test_client.post("/health").status_code == 405

    def test_predict_returns_503_when_model_missing(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """
        Without a model the API must fail fast with 503, not 500.

        503 tells a load balancer to route elsewhere; 500 does not.
        """
        monkeypatch.setattr("app.main.model", None)
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_batch_returns_503_when_model_missing(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """The batch endpoint applies the same availability rule."""
        monkeypatch.setattr("app.main.model", None)
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 503

    def test_predict_returns_500_when_model_raises(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """An unexpected model failure is contained and surfaced as a 500."""

        class ExplodingModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise RuntimeError("inference exploded")

        monkeypatch.setattr("app.main.model", ExplodingModel())
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 500
        assert "inference exploded" in response.json()["detail"]

    def test_batch_returns_500_when_model_raises(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """The batch endpoint contains model failures the same way."""

        class ExplodingModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise RuntimeError("batch inference exploded")

        monkeypatch.setattr("app.main.model", ExplodingModel())
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 500

    def test_predict_returns_503_when_model_reports_not_loaded(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """
        The model can pass is_loaded() and still refuse mid-request.

        That race (an artifact swapped out under a live worker) maps to 503, so
        the caller retries elsewhere instead of seeing a 500.
        """

        class VanishingModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise ModelNotLoadedError("Model not loaded")

        monkeypatch.setattr("app.main.model", VanishingModel())
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_batch_returns_503_when_model_reports_not_loaded(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """
        A ModelNotLoadedError raised mid-batch maps to 503, not 500.

        The distinction matters to a load balancer: 503 means "try another
        instance", 500 means "this request is broken".
        """

        class VanishingModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise ModelNotLoadedError("Model not loaded")

        monkeypatch.setattr("app.main.model", VanishingModel())
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 503

    def test_predict_returns_422_when_model_rejects_ids(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """A ValueError from the model is a client error, reported as 422."""

        class PickyModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise ValueError("user_id and movie_id must be non-empty strings")

        monkeypatch.setattr("app.main.model", PickyModel())
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422

    def test_batch_returns_422_when_model_rejects_ids(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_batch_request: Dict[str, Any],
    ) -> None:
        """The batch endpoint applies the same mapping."""

        class PickyModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                raise ValueError("bad id")

        monkeypatch.setattr("app.main.model", PickyModel())
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 422

    def test_cors_headers_are_present(self, test_client: TestClient) -> None:
        """CORS middleware is wired up for browser clients."""
        response = test_client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "*"


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, test_client: TestClient) -> None:
        """Test that model info endpoint returns 200."""
        response = test_client.get("/model/info")
        assert response.status_code == 200

    # =========================================================================
    # TODO 8: Model Info Tests
    # =========================================================================

    def test_model_info_has_version(self, test_client: TestClient) -> None:
        """The deployed model version is discoverable at runtime."""
        data = test_client.get("/model/info").json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_model_info_has_is_loaded(self, test_client: TestClient) -> None:
        """The endpoint reports load state as a boolean."""
        data = test_client.get("/model/info").json()
        assert isinstance(data["is_loaded"], bool)

    def test_model_info_reports_model_type(self, test_client: TestClient) -> None:
        """The algorithm family is documented in the response."""
        data = test_client.get("/model/info").json()
        assert "SVD" in data["model_type"]

    def test_model_info_agrees_with_health(self, test_client: TestClient) -> None:
        """/model/info and /health must not report different load states."""
        info = test_client.get("/model/info").json()
        health = test_client.get("/health").json()
        assert info["is_loaded"] == health["model_loaded"]

    def test_model_info_reports_cache_state(self, test_client: TestClient) -> None:
        """Whether the cache is switched on is part of the deployment's identity."""
        assert isinstance(test_client.get("/model/info").json()["cache_enabled"], bool)

    def test_model_info_exposes_training_metrics(self, test_client: TestClient) -> None:
        """
        The RMSE/MAE recorded at training time are served alongside the version.

        This is what makes a running instance auditable: you can ask a live
        service how good the model it is serving actually was.
        """
        data = test_client.get("/model/info").json()
        if "metrics" not in data:
            pytest.skip("models/model_metadata.json not present")
        assert set(data["metrics"]) >= {"rmse", "mae", "cv_folds"}
        assert 0.0 < data["metrics"]["rmse"] < 2.0
        assert data["training_set"]["dataset"] == "MovieLens 100K"

    def test_model_info_omits_absent_metadata(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        With no metadata file the optional fields are omitted entirely.

        `response_model_exclude_none=True` means clients see a missing key
        rather than an explicit null they have to special-case.
        """
        monkeypatch.setattr("app.main._load_model_metadata", dict)
        data = test_client.get("/model/info").json()
        assert "metrics" not in data
        assert "training_set" not in data
        assert data["model_version"]

    def test_corrupted_metadata_file_is_ignored(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        Unreadable metadata degrades to "no metadata", it does not 500.

        Metadata is descriptive; a broken file must not take the API down.
        """
        broken = tmp_path / "svd_model.pkl"
        broken.write_bytes(b"")
        (tmp_path / "model_metadata.json").write_text("{ not json", encoding="utf-8")
        monkeypatch.setattr("app.main.MODEL_PATH", str(broken))

        response = test_client.get("/model/info")
        assert response.status_code == 200
        assert "metrics" not in response.json()


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
