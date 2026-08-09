"""
FastAPI application for Movie Rating Prediction.
"""

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import logging

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics as app_metrics
from app.cache import PredictionCache, get_cache
from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    METRICS_ENABLED,
    MODEL_VERSION,
    MODEL_PATH,
)
from app.model import MovieRatingModel, ModelNotLoadedError
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    RootResponse,
    ErrorResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Initialize FastAPI app
# =============================================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_TYPE = "SVD (Collaborative Filtering)"


@app.middleware("http")
async def observe_request(request: Request, call_next):
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


# =============================================================================
# Load model and cache at startup
# =============================================================================
model: Optional[MovieRatingModel] = None
cache: Optional[PredictionCache] = None


def _load_model_metadata() -> dict:
    """Read training metadata written next to the model file, if present."""
    metadata_path = Path(MODEL_PATH).with_name("model_metadata.json")
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _init_model() -> None:
    """Load the model once, tolerating failure so /health can report it."""
    global model
    if model is not None:
        return
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Model stays None; health check will report unhealthy.

    if METRICS_ENABLED:
        app_metrics.set_model_state(
            version=MODEL_VERSION,
            model_type=MODEL_TYPE,
            loaded=model is not None and model.is_loaded(),
            metrics=_load_model_metadata().get("metrics"),
        )


def _init_cache() -> None:
    """Connect the optional prediction cache. Never fatal."""
    global cache
    if cache is None:
        cache = get_cache()
    if METRICS_ENABLED:
        app_metrics.CACHE_CONNECTED.set(1 if cache.is_connected() else 0)


# Load eagerly at import time: uvicorn workers and TestClient instances that do
# not run the lifespan both get a ready model this way.
_init_model()
_init_cache()


@app.on_event("startup")
async def startup_event():
    """Load model and cache when application starts."""
    _init_model()
    _init_cache()


# =============================================================================
# Health Check Endpoint (PROVIDED - DO NOT MODIFY)
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded()
    )


# =============================================================================
# Prediction endpoint
# =============================================================================
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict a rating for one user-movie pair",
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(request: PredictionRequest):
    """
    Predict the rating a user would give to a movie, on the MovieLens 1-5 scale.

    Unknown user or movie IDs are **not** an error: the model falls back to the
    global mean rating, so cold-start requests still get an answer.

    Results are served from Redis when the cache is enabled and warm.

    Errors: `422` for an invalid body, `503` if the model is unavailable,
    `500` if the prediction itself fails.
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    rating = cache.get(request.user_id, request.movie_id) if cache else None
    cache_hit = rating is not None

    if not cache_hit:
        try:
            rating = model.predict(request.user_id, request.movie_id)
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        if cache:
            cache.set(request.user_id, request.movie_id, rating)

    if METRICS_ENABLED:
        app_metrics.record_cache(hit=cache_hit)
        app_metrics.PREDICTIONS.labels(endpoint="/predict").inc()

    return PredictionResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        predicted_rating=rating,
        model_version=MODEL_VERSION,
    )


# =============================================================================
# Batch prediction endpoint (BONUS)
# =============================================================================
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict ratings for up to 100 pairs",
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs in one round trip.

    Results come back in request order. Batches are capped at 100 pairs to keep
    tail latency bounded; a larger batch is rejected with `422`.

    Each pair is looked up in the cache individually, so a partially warm batch
    only computes the pairs it has to.
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        ratings = []
        for item in request.predictions:
            rating = cache.get(item.user_id, item.movie_id) if cache else None
            hit = rating is not None
            if not hit:
                rating = model.predict(item.user_id, item.movie_id)
                if cache:
                    cache.set(item.user_id, item.movie_id, rating)
            if METRICS_ENABLED:
                app_metrics.record_cache(hit=hit)
            ratings.append(rating)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if METRICS_ENABLED:
        app_metrics.PREDICTIONS.labels(endpoint="/predict/batch").inc(len(ratings))

    results = [
        PredictionResponse(
            user_id=item.user_id,
            movie_id=item.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION,
        )
        for item, rating in zip(request.predictions, ratings)
    ]
    return BatchPredictionResponse(predictions=results, total_count=len(results))


# =============================================================================
# Root endpoint
# =============================================================================
@app.get("/", response_model=RootResponse, tags=["Info"], summary="API metadata")
async def root():
    """Root endpoint with API information and links to the docs and health check."""
    return RootResponse(
        name=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs="/docs",
        health="/health",
    )


# =============================================================================
# Model info endpoint
# =============================================================================
@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    response_model_exclude_none=True,
    tags=["Info"],
    summary="Model version and training metrics",
)
async def model_info():
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
        metrics=metadata.get("metrics"),
        trained_at=metadata.get("trained_at"),
        training_set=metadata.get("training_set"),
    )


# =============================================================================
# Prometheus metrics endpoint
# =============================================================================
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
async def prometheus_metrics():
    """
    Metrics for Prometheus to scrape: request counts and latency histograms,
    prediction and cache-hit counters, model load state, and the deployed
    model's training scores.
    """
    # Refresh the gauges that describe current state rather than accumulate.
    app_metrics.MODEL_LOADED.set(1 if model is not None and model.is_loaded() else 0)
    app_metrics.CACHE_CONNECTED.set(1 if cache is not None and cache.is_connected() else 0)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Run with uvicorn (for development)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
