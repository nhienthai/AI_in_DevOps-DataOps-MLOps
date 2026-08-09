"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


def _validate_id(value: str, field_name: str) -> str:
    """Trim an ID and reject blank ones, so every 422 comes from validation."""
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{field_name} must not be blank")
    return trimmed


# =============================================================================
# Prediction schemas
# =============================================================================

class PredictionRequest(BaseModel):
    """Request schema for the /predict endpoint."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="MovieLens user ID, as a string.",
        examples=["196"],
    )
    movie_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="MovieLens movie (item) ID, as a string.",
        examples=["242"],
    )

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        return _validate_id(v, "user_id")

    @field_validator("movie_id")
    @classmethod
    def _check_movie_id(cls, v: str) -> str:
        return _validate_id(v, "movie_id")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "196", "movie_id": "242"}]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    user_id: str = Field(..., description="Echo of the requested user ID.", examples=["196"])
    movie_id: str = Field(..., description="Echo of the requested movie ID.", examples=["242"])
    predicted_rating: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Predicted rating on the MovieLens 1-5 scale, rounded to 2 decimals.",
        examples=[3.8],
    )
    model_version: str = Field(
        ...,
        description="Version of the model that served this prediction.",
        examples=["1.0.0"],
    )

    # "model_version" is not a Pydantic-reserved name here; opt out of the warning.
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "196",
                    "movie_id": "242",
                    "predicted_rating": 3.8,
                    "model_version": "1.0.0",
                }
            ]
        },
    }


# =============================================================================
# Health schema
# =============================================================================

class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(
        ...,
        description='"healthy" when the model is ready to serve, otherwise "unhealthy".',
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model file was loaded successfully.",
        examples=[True],
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "examples": [{"status": "healthy", "model_loaded": True}]
        },
    }


# =============================================================================
# Batch prediction schemas (BONUS)
# =============================================================================

class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""

    user_id: str = Field(..., min_length=1, max_length=64, examples=["196"])
    movie_id: str = Field(..., min_length=1, max_length=64, examples=["242"])

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        return _validate_id(v, "user_id")

    @field_validator("movie_id")
    @classmethod
    def _check_movie_id(cls, v: str) -> str:
        return _validate_id(v, "movie_id")


class BatchPredictionRequest(BaseModel):
    """Request schema for the /predict/batch endpoint."""

    predictions: List[PredictionItem] = Field(
        ...,
        max_length=100,
        description="User-movie pairs to score. At most 100 per request.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predictions": [
                        {"user_id": "196", "movie_id": "242"},
                        {"user_id": "186", "movie_id": "302"},
                    ]
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response schema for the /predict/batch endpoint."""

    predictions: List[PredictionResponse] = Field(
        ..., description="Predicted ratings, in the same order as the request."
    )
    total_count: int = Field(..., description="Number of predictions returned.", examples=[2])


# =============================================================================
# Info schemas
# =============================================================================

class RootResponse(BaseModel):
    """Response schema for the / endpoint."""

    name: str = Field(..., examples=["Movie Rating Prediction API"])
    version: str = Field(..., examples=["1.0.0"])
    description: str = Field(
        ..., examples=["API for predicting movie ratings using collaborative filtering"]
    )
    docs: str = Field(..., description="Path to the Swagger UI.", examples=["/docs"])
    health: str = Field(..., description="Path to the health check.", examples=["/health"])


class ModelMetrics(BaseModel):
    """Cross-validation metrics recorded at training time."""

    rmse: float = Field(..., description="Mean RMSE across CV folds.", examples=[0.9353])
    mae: float = Field(..., description="Mean MAE across CV folds.", examples=[0.7374])
    cv_folds: int = Field(..., description="Number of cross-validation folds.", examples=[5])


class TrainingSetInfo(BaseModel):
    """Shape of the data the model was trained on."""

    dataset: str = Field(..., examples=["MovieLens 100K"])
    n_users: int = Field(..., examples=[943])
    n_items: int = Field(..., examples=[1682])
    n_ratings: int = Field(..., examples=[100000])


class ModelInfoResponse(BaseModel):
    """Response schema for the /model/info endpoint."""

    model_version: str = Field(..., examples=["1.0.0"])
    model_type: str = Field(..., examples=["SVD (Collaborative Filtering)"])
    is_loaded: bool = Field(
        ..., description="Whether the model is currently serving.", examples=[True]
    )
    cache_enabled: bool = Field(
        False,
        description="Whether the Redis prediction cache is switched on.",
        examples=[True],
    )
    metrics: Optional[ModelMetrics] = Field(
        None, description="Absent when models/model_metadata.json is missing."
    )
    trained_at: Optional[str] = Field(
        None, description="UTC timestamp of the training run.", examples=["2026-08-08T13:06:16+00:00"]
    )
    training_set: Optional[TrainingSetInfo] = None

    model_config = {"protected_namespaces": ()}


# =============================================================================
# Error schema
# =============================================================================

class ErrorResponse(BaseModel):
    """
    Body returned for errors raised explicitly by the API (503, 500).

    Request-validation failures (422) use FastAPI's own `HTTPValidationError`
    shape instead, where `detail` is a list of per-field error objects.
    """

    detail: str = Field(..., examples=["Model not loaded"])
