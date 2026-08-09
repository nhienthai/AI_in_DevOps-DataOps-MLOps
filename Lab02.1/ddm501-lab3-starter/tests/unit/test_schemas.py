"""
Unit tests for Pydantic schemas.

Schemas are the API's contract. These tests pin down both directions: what the
API must accept, and - more importantly - what it must reject before any of it
reaches the model.

Run tests:
    pytest tests/unit/test_schemas.py -v
"""

from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from app.config import MAX_ID_LENGTH
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelMetrics,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
    TrainingSetInfo,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    # =========================================================================
    # Valid Input Tests (PROVIDED)
    # =========================================================================

    def test_valid_request(self) -> None:
        """Test that valid request passes validation."""
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self) -> None:
        """Test numeric string IDs are valid."""
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    # =========================================================================
    # TODO 1: Missing Field Tests
    # =========================================================================

    def test_missing_user_id_raises_error(self) -> None:
        """user_id is required."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(movie_id="242")  # type: ignore[call-arg]
        assert "user_id" in str(exc_info.value)

    def test_missing_movie_id_raises_error(self) -> None:
        """movie_id is required."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(user_id="196")  # type: ignore[call-arg]
        assert "movie_id" in str(exc_info.value)

    def test_missing_both_fields_raises_error(self) -> None:
        """Both fields missing reports two distinct errors."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest()  # type: ignore[call-arg]
        assert len(exc_info.value.errors()) == 2

    # =========================================================================
    # TODO 2: Empty/Invalid Input Tests
    # =========================================================================

    def test_empty_user_id_raises_error(self) -> None:
        """An empty user_id violates min_length."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_empty_movie_id_raises_error(self) -> None:
        """An empty movie_id violates min_length."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id="")

    def test_whitespace_only_user_id_raises_error(self) -> None:
        """Whitespace-only IDs are caught by the custom field validator."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_whitespace_only_movie_id_raises_error(self) -> None:
        """The validator is registered for movie_id too, not only user_id."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id="\t\n ")

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """Valid IDs padded with whitespace are normalised, not rejected."""
        request = PredictionRequest(user_id="  196  ", movie_id=" 242 ")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_none_values_raise_error(self) -> None:
        """None is not a string and must be rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id="242")  # type: ignore[arg-type]

    def test_too_long_user_id_raises_error(self) -> None:
        """One character past MAX_ID_LENGTH is rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="1" * (MAX_ID_LENGTH + 1), movie_id="242")

    def test_max_length_user_id_is_accepted(self) -> None:
        """Exactly MAX_ID_LENGTH characters is still valid (boundary)."""
        request = PredictionRequest(user_id="1" * MAX_ID_LENGTH, movie_id="242")
        assert len(request.user_id) == MAX_ID_LENGTH

    def test_batch_item_shares_the_same_id_limit(self) -> None:
        """
        /predict and /predict/batch must enforce one identical ID contract.

        Both schemas read MAX_ID_LENGTH from config, so the two endpoints cannot
        drift apart the way they did between Lab 1 (64) and the Lab 3 starter (50).
        """
        PredictionItem(user_id="1" * MAX_ID_LENGTH, movie_id="242")
        with pytest.raises(ValidationError):
            PredictionItem(user_id="1" * (MAX_ID_LENGTH + 1), movie_id="242")

    # =========================================================================
    # TODO 3: Type Validation Tests
    # =========================================================================

    def test_integer_user_id_is_rejected_in_strict_str_field(self) -> None:
        """
        Pydantic v2 does NOT coerce int -> str by default.

        This is worth pinning: a client sending {"user_id": 196} gets a 422
        rather than silently being treated as the string "196".
        """
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=196, movie_id="242")  # type: ignore[arg-type]

    def test_list_user_id_is_rejected(self) -> None:
        """Structured values are not acceptable IDs."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=["196"], movie_id="242")  # type: ignore[arg-type]

    def test_extra_fields_are_ignored(self) -> None:
        """Unknown keys do not break the request (default pydantic behaviour)."""
        request = PredictionRequest(user_id="196", movie_id="242", unexpected="x")
        assert request.user_id == "196"


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    # =========================================================================
    # TODO 4: Response Validation Tests
    # =========================================================================

    def test_valid_response(self) -> None:
        """A well-formed response validates and keeps its values."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        assert response.predicted_rating == 3.5
        assert response.model_version == "1.0.0"

    def test_rating_below_minimum_raises_error(self) -> None:
        """The API must never emit a rating below 1.0."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=0.5, model_version="1.0.0"
            )

    def test_rating_above_maximum_raises_error(self) -> None:
        """The API must never emit a rating above 5.0."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=5.5, model_version="1.0.0"
            )

    @pytest.mark.parametrize("rating", [1.0, 5.0])
    def test_rating_at_boundaries(self, rating: float) -> None:
        """Both boundaries are inclusive (ge/le, not gt/lt)."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=rating, model_version="1.0.0"
        )
        assert response.predicted_rating == rating

    def test_integer_rating_is_coerced_to_float(self) -> None:
        """int -> float is a lossless widening, so pydantic allows it."""
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=4, model_version="1.0.0"
        )
        assert isinstance(response.predicted_rating, float)

    def test_missing_model_version_raises_error(self) -> None:
        """Responses must always carry the model version for traceability."""
        with pytest.raises(ValidationError):
            PredictionResponse(  # type: ignore[call-arg]
                user_id="196", movie_id="242", predicted_rating=3.5
            )


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    # =========================================================================
    # TODO 5: Health Response Tests
    # =========================================================================

    def test_valid_health_response(self) -> None:
        """A healthy payload validates."""
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True

    @pytest.mark.parametrize(
        ("status", "model_loaded"),
        [("healthy", True), ("unhealthy", False), ("degraded", True)],
    )
    def test_health_response_status_types(self, status: str, model_loaded: bool) -> None:
        """status is a free-form string; the schema does not constrain vocabulary."""
        response = HealthResponse(status=status, model_loaded=model_loaded)
        assert response.status == status
        assert response.model_loaded == model_loaded

    def test_health_response_requires_model_loaded(self) -> None:
        """model_loaded is mandatory - an absent flag would hide an outage."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy")  # type: ignore[call-arg]

    def test_non_boolean_model_loaded_is_rejected(self) -> None:
        """A non-boolean flag must not slip through."""
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", model_loaded="maybe")  # type: ignore[arg-type]


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    # =========================================================================
    # TODO 6: Batch Request Tests
    # =========================================================================

    def test_valid_batch_request(self, sample_batch_request: Dict[str, Any]) -> None:
        """A batch of three items validates and keeps its order."""
        request = BatchPredictionRequest(**sample_batch_request)
        assert len(request.predictions) == 3
        assert request.predictions[0].user_id == "196"

    def test_empty_predictions_list_raises_error(self) -> None:
        """min_length=1 rejects an empty batch."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_missing_predictions_key_raises_error(self) -> None:
        """The predictions key is required."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest()  # type: ignore[call-arg]

    def test_single_prediction_is_accepted(self) -> None:
        """One item is the smallest valid batch (boundary)."""
        request = BatchPredictionRequest(predictions=[PredictionItem(user_id="1", movie_id="2")])
        assert len(request.predictions) == 1

    def test_maximum_batch_size_is_accepted(self) -> None:
        """Exactly 100 items is still valid (boundary)."""
        items: List[Dict[str, str]] = [{"user_id": str(i), "movie_id": str(i)} for i in range(100)]
        request = BatchPredictionRequest(predictions=items)  # type: ignore[arg-type]
        assert len(request.predictions) == 100

    def test_too_many_predictions_raises_error(self) -> None:
        """max_length=100 protects the service from unbounded batches."""
        items = [{"user_id": str(i), "movie_id": str(i)} for i in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)  # type: ignore[arg-type]

    def test_invalid_item_inside_batch_raises_error(self) -> None:
        """One bad item invalidates the whole batch."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                predictions=[  # type: ignore[arg-type]
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "", "movie_id": "302"},
                ]
            )


class TestBatchPredictionResponse:
    """Tests for BatchPredictionResponse schema."""

    def test_valid_batch_response(self) -> None:
        """total_count and the payload length are both carried explicitly."""
        item = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        response = BatchPredictionResponse(predictions=[item], total_count=1)
        assert response.total_count == 1
        assert len(response.predictions) == 1


class TestRootResponse:
    """Tests for RootResponse schema (ported from Lab 1)."""

    def test_valid_root_response(self) -> None:
        """The landing payload is a typed contract, not a loose dict."""
        response = RootResponse(
            name="Movie Rating Prediction API",
            version="1.0.0",
            description="desc",
            docs="/docs",
            health="/health",
        )
        assert response.docs == "/docs"

    def test_root_response_requires_every_field(self) -> None:
        """Dropping a documented link is a contract break, not a silent omission."""
        with pytest.raises(ValidationError):
            RootResponse(  # type: ignore[call-arg]
                name="x", version="1.0.0", description="d", docs="/docs"
            )


class TestModelInfoResponse:
    """Tests for ModelInfoResponse and its nested schemas (ported from Lab 1)."""

    def test_minimal_model_info(self) -> None:
        """Metrics and training-set details are optional - absent metadata is valid."""
        response = ModelInfoResponse(
            model_version="1.0.0", model_type="SVD (Collaborative Filtering)", is_loaded=True
        )
        assert response.metrics is None
        assert response.training_set is None
        assert response.cache_enabled is False

    def test_full_model_info(self) -> None:
        """When metadata exists it is parsed into typed nested models."""
        response = ModelInfoResponse(
            model_version="1.0.0",
            model_type="SVD (Collaborative Filtering)",
            is_loaded=True,
            cache_enabled=True,
            backend="local",
            metrics={"rmse": 0.944, "mae": 0.745, "cv_folds": 3},  # type: ignore[arg-type]
            trained_at="2026-08-09T00:00:00+00:00",
            training_set={  # type: ignore[arg-type]
                "dataset": "MovieLens 100K",
                "n_users": 943,
                "n_items": 1682,
                "n_ratings": 100000,
            },
        )
        assert isinstance(response.metrics, ModelMetrics)
        assert isinstance(response.training_set, TrainingSetInfo)
        assert response.metrics.cv_folds == 3
        assert response.training_set.n_users == 943

    def test_malformed_metrics_are_rejected(self) -> None:
        """A metadata file with the wrong shape fails loudly at the boundary."""
        with pytest.raises(ValidationError):
            ModelInfoResponse(
                model_version="1.0.0",
                model_type="SVD",
                is_loaded=True,
                metrics={"rmse": "not-a-number"},  # type: ignore[arg-type]
            )

    def test_model_prefixed_fields_do_not_warn(self) -> None:
        """
        `model_version` / `model_type` sit in pydantic's protected `model_`
        namespace; the schema opts out rather than renaming a public field.
        """
        assert ModelInfoResponse.model_config.get("protected_namespaces") == ()


class TestHealthResponseCacheField:
    """The cache_connected field added when the Lab 1 cache was ported."""

    def test_cache_connected_defaults_to_false(self) -> None:
        """Health stays constructible without knowing about the cache."""
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.cache_connected is False

    def test_cache_connected_can_be_set(self) -> None:
        """A reachable cache is reported explicitly."""
        response = HealthResponse(status="healthy", model_loaded=True, cache_connected=True)
        assert response.cache_connected is True


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response_defaults_error_code(self) -> None:
        """error_code falls back to UNKNOWN_ERROR when not supplied."""
        response = ErrorResponse(detail="something went wrong")
        assert response.error_code == "UNKNOWN_ERROR"

    def test_error_response_accepts_explicit_code(self) -> None:
        """An explicit machine-readable code overrides the default."""
        response = ErrorResponse(detail="model missing", error_code="MODEL_NOT_LOADED")
        assert response.error_code == "MODEL_NOT_LOADED"


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
