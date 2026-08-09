"""
FastAPI application for Movie Rating Prediction.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics as app_metrics
from app.cache import PredictionCache, get_cache
from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    METRICS_ENABLED,
    MODEL_PATH,
    MODEL_TYPE,
    MODEL_VERSION,
)
from app.model import ModelNotLoadedError, MovieRatingModel
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals populated by the lifespan handler at startup.
model: Optional[MovieRatingModel] = None
cache: Optional[PredictionCache] = None

# Error responses documented in the OpenAPI schema for the prediction routes.
PREDICTION_ERROR_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    503: {"model": ErrorResponse, "description": "Model not loaded"},
    500: {"model": ErrorResponse, "description": "Prediction failed"},
}


def _load_model_metadata() -> Dict[str, Any]:
    """Read training metadata written next to the model file, if present."""
    metadata_path = Path(MODEL_PATH).with_name("model_metadata.json")
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model and connect the cache at startup; release both on shutdown."""
    global model, cache

    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:  # pragma: no cover - defensive, keeps /health serving
        logger.error(f"Failed to load model: {e}")
        model = None

    # The cache never raises: a missing Redis degrades to model-only serving.
    cache = get_cache()

    if METRICS_ENABLED:
        app_metrics.set_model_state(
            version=MODEL_VERSION,
            model_type=MODEL_TYPE,
            loaded=model is not None and model.is_loaded(),
            metrics=_load_model_metadata().get("metrics"),
        )
        app_metrics.CACHE_CONNECTED.set(1 if cache.is_connected() else 0)

    yield

    model = None
    cache = None


# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Expose per-request latency as a header and as Prometheus metrics."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"

    if METRICS_ENABLED:
        # Label on the route template ("/predict"), never the raw URL, or every
        # distinct path would become its own time series.
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        if path and path != "/metrics":
            app_metrics.record_request(request.method, path, response.status_code, elapsed)

    return response


def _require_model() -> MovieRatingModel:
    """Return the loaded model or raise a 503 if it is unavailable."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _predict_cached(active_model: MovieRatingModel, user_id: str, movie_id: str) -> float:
    """Predict one pair, consulting the cache first and recording the outcome."""
    cached: Optional[float] = cache.get(user_id, movie_id) if cache else None

    if cached is None:
        rating = active_model.predict(user_id, movie_id)
        if cache:
            cache.set(user_id, movie_id, rating)
    else:
        rating = cached

    if METRICS_ENABLED:
        app_metrics.record_cache(hit=cached is not None)

    return rating


@app.get("/", response_model=RootResponse, tags=["Info"], summary="API metadata")
async def root() -> RootResponse:
    """Root endpoint with API information and links to the docs and health check."""
    return RootResponse(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs="/docs",
        health="/health",
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Reports whether the model is loaded and whether the optional prediction
    cache is reachable. The cache being down does NOT make the API unhealthy -
    it only removes the speedup.
    """
    loaded = model is not None and model.is_loaded()
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        cache_connected=cache is not None and cache.is_connected(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict a rating for one user-movie pair",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict the rating a user would give to a movie, on the MovieLens 1-5 scale.

    Unknown user or movie IDs are **not** an error: the model falls back to the
    global mean rating, so cold-start requests still get an answer.

    Results are served from Redis when the cache is enabled and warm.

    Errors: `422` for an invalid body, `503` if the model is unavailable,
    `500` if the prediction itself fails.
    """
    active_model = _require_model()

    try:
        rating = _predict_cached(active_model, request.user_id, request.movie_id)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if METRICS_ENABLED:
        app_metrics.record_predictions("/predict")

    return PredictionResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        predicted_rating=rating,
        model_version=MODEL_VERSION,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict ratings for up to 100 pairs",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict movie ratings for multiple user-movie pairs in one round trip.

    Results come back in request order. Batches are capped at 100 pairs to keep
    tail latency bounded; a larger batch is rejected with `422`.

    Each pair is looked up in the cache individually, so a partially warm batch
    only computes the pairs it has to.
    """
    active_model = _require_model()

    try:
        ratings = [
            _predict_cached(active_model, item.user_id, item.movie_id)
            for item in request.predictions
        ]
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if METRICS_ENABLED:
        app_metrics.record_predictions("/predict/batch", len(ratings))

    results: List[PredictionResponse] = [
        PredictionResponse(
            user_id=item.user_id,
            movie_id=item.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION,
        )
        for item, rating in zip(request.predictions, ratings)
    ]
    return BatchPredictionResponse(predictions=results, total_count=len(results))


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    response_model_exclude_none=True,
    tags=["Info"],
    summary="Model version and training metrics",
)
async def model_info() -> ModelInfoResponse:
    """
    Model version, algorithm, load state, and the cross-validation metrics
    recorded by `scripts/train_model.py`.

    Metric fields are omitted when `models/model_metadata.json` is absent.
    """
    metadata = _load_model_metadata()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        is_loaded=model is not None and model.is_loaded(),
        cache_enabled=cache is not None and cache.enabled,
        backend=metadata.get("backend"),
        metrics=metadata.get("metrics"),
        trained_at=metadata.get("trained_at"),
        training_set=metadata.get("training_set"),
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus metrics",
    response_class=Response,
    responses={
        200: {
            "content": {CONTENT_TYPE_LATEST: {}},
            "description": "Metrics in the Prometheus text exposition format.",
        }
    },
)
async def prometheus_metrics() -> Response:
    """
    Metrics for Prometheus to scrape: request counts and latency histograms,
    prediction and cache-hit counters, model load state, and the deployed
    model's training scores.
    """
    # Refresh the gauges that describe current state rather than accumulate.
    app_metrics.MODEL_LOADED.set(1 if model is not None and model.is_loaded() else 0)
    app_metrics.CACHE_CONNECTED.set(1 if cache is not None and cache.is_connected() else 0)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
