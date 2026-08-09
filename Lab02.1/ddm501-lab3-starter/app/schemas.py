"""
Pydantic schemas for request/response validation.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_ID_LENGTH, MAX_RATING, MIN_RATING

# `model_version` / `model_loaded` / `model_type` collide with pydantic v2's
# protected `model_` namespace, which emits a UserWarning on every import. The
# names are part of the published API contract, so the namespace is opened
# rather than the fields renamed.
ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


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
        max_length=MAX_ID_LENGTH,
        description="MovieLens user ID, as a string.",
        examples=["196"],
    )
    movie_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_ID_LENGTH,
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

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"user_id": "196", "movie_id": "242"}]}
    )


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    model_config = ALLOW_MODEL_PREFIX

    user_id: str = Field(..., description="Echo of the requested user ID.", examples=["196"])
    movie_id: str = Field(..., description="Echo of the requested movie ID.", examples=["242"])
    predicted_rating: float = Field(
        ...,
        ge=MIN_RATING,
        le=MAX_RATING,
        description="Predicted rating on the MovieLens 1-5 scale, rounded to 2 decimals.",
        examples=[3.8],
    )
    model_version: str = Field(
        ...,
        description="Version of the model that served this prediction.",
        examples=["1.0.0"],
    )


class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""

    user_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, examples=["196"])
    movie_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, examples=["242"])

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
        min_length=1,
        max_length=100,
        description="User-movie pairs to score. Between 1 and 100 per request.",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for the /predict/batch endpoint."""

    predictions: List[PredictionResponse] = Field(
        ..., description="Predicted ratings, in the same order as the request."
    )
    total_count: int = Field(..., description="Number of predictions returned.", examples=[2])


# =============================================================================
# Health schema
# =============================================================================


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    model_config = ALLOW_MODEL_PREFIX

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
    cache_connected: bool = Field(
        False,
        description="Whether the Redis prediction cache is currently reachable.",
        examples=[False],
    )


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

    model_config = ALLOW_MODEL_PREFIX

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
    backend: Optional[str] = Field(
        None, description="Training backend that produced the artifact.", examples=["surprise"]
    )
    metrics: Optional[ModelMetrics] = Field(
        None, description="Absent when models/model_metadata.json is missing."
    )
    trained_at: Optional[str] = Field(
        None,
        description="UTC timestamp of the training run.",
        examples=["2026-08-08T13:06:16+00:00"],
    )
    training_set: Optional[TrainingSetInfo] = None


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
    error_code: str = Field("UNKNOWN_ERROR", examples=["MODEL_NOT_LOADED"])
