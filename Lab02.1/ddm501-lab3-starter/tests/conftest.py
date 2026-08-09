"""
Shared pytest fixtures for all tests.

This file is automatically loaded by pytest and provides
fixtures that can be used across all test modules.
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import MovieRatingModel

# =============================================================================
# API Client Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_client() -> Iterator[TestClient]:
    """
    Create a test client for API tests.

    Scope: session - created once for all tests.

    NOTE: the client is used as a context manager so Starlette runs the
    application lifespan. Without it the startup handler never fires and the
    model global stays ``None``, which would make every /predict call return 503.
    """
    with TestClient(app) as client:
        yield client


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def api_client(test_client: TestClient) -> TestClient:
    """
    Test client guaranteed to have a model loaded.

    Endpoints that need the model return 503 without it, so tests that assert on
    predictions skip rather than fail when the artifact was never built.
    """
    if not test_client.get("/health").json()["model_loaded"]:
        pytest.skip("Model file not found. Run train_model.py first.")
    return test_client


@pytest.fixture(scope="session")
def trained_model() -> MovieRatingModel:
    """
    Load model once for all tests.

    Scope: session - model is loaded once and reused
    """
    try:
        return MovieRatingModel()
    except FileNotFoundError:
        pytest.skip("Model file not found. Run train_model.py first.")


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prediction_request() -> Dict[str, str]:
    """Sample valid prediction request."""
    return {"user_id": "196", "movie_id": "242"}


@pytest.fixture
def sample_batch_request() -> Dict[str, List[Dict[str, str]]]:
    """Sample batch prediction request."""
    return {
        "predictions": [
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "186", "movie_id": "302"},
            {"user_id": "22", "movie_id": "377"},
        ]
    }


@pytest.fixture
def sample_ratings() -> List[Dict[str, Any]]:
    """Sample ratings data for data quality tests."""
    return [
        {"user_id": "1", "movie_id": "10", "rating": 4.0},
        {"user_id": "1", "movie_id": "20", "rating": 3.5},
        {"user_id": "2", "movie_id": "10", "rating": 5.0},
        {"user_id": "2", "movie_id": "30", "rating": 2.0},
        {"user_id": "3", "movie_id": "10", "rating": 3.0},
        {"user_id": "3", "movie_id": "20", "rating": 4.5},
        {"user_id": "3", "movie_id": "30", "rating": 1.0},
    ]


@pytest.fixture
def ratings_dataframe(sample_ratings: List[Dict[str, Any]]) -> pd.DataFrame:
    """Sample ratings as a DataFrame, the shape data validators expect."""
    return pd.DataFrame(sample_ratings)


@pytest.fixture
def corrupted_ratings() -> List[Dict[str, Any]]:
    """
    Deliberately broken ratings used to prove the validators actually fail.

    A data test that only ever sees clean data cannot tell you whether it works.
    """
    return [
        {"user_id": "1", "movie_id": "10", "rating": 7.0},  # out of range
        {"user_id": "2", "movie_id": "20", "rating": None},  # missing value
        {"user_id": "", "movie_id": "30", "rating": 3.0},  # empty id
        {"user_id": "4", "movie_id": "40", "rating": "four"},  # wrong type
    ]


@pytest.fixture
def invalid_prediction_requests() -> List[Dict[str, str]]:
    """Collection of invalid prediction requests for testing validation."""
    return [
        {},  # Empty
        {"user_id": "196"},  # Missing movie_id
        {"movie_id": "242"},  # Missing user_id
        {"user_id": "", "movie_id": "242"},  # Empty user_id
        {"user_id": "196", "movie_id": ""},  # Empty movie_id
        {"user_id": "   ", "movie_id": "242"},  # Whitespace user_id
    ]


# =============================================================================
# Real Dataset Fixture (MovieLens 100K)
# =============================================================================


def _find_movielens_file() -> Path | None:
    """Locate the raw MovieLens 100K ratings file downloaded by scikit-surprise."""
    candidates = [
        Path(os.getenv("SURPRISE_DATA_FOLDER", Path.home() / ".surprise_data")),
        Path.home() / ".surprise_data",
    ]
    for base in candidates:
        path = base / "ml-100k" / "ml-100k" / "u.data"
        if path.exists():
            return path
    return None


@pytest.fixture(scope="session")
def movielens_ratings() -> pd.DataFrame:
    """
    Raw MovieLens 100K ratings as a DataFrame.

    Skipped when the dataset has not been downloaded yet (i.e. train_model.py
    has never run on this machine), so the suite stays runnable offline.
    """
    path = _find_movielens_file()
    if path is None:
        pytest.skip("MovieLens 100K not downloaded. Run scripts/train_model.py first.")
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": str, "movie_id": str, "rating": float, "timestamp": int},
    )


@pytest.fixture(scope="session")
def evaluation_sample(movielens_ratings: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """
    A fixed random sample of real ``(user_id, movie_id, actual_rating)`` rows.

    Behavioural quality gates need a statistically meaningful sample; a handful
    of hand-picked pairs measures luck, not the model. The seed is fixed so the
    gate is reproducible across runs and machines.
    """
    sample = movielens_ratings.sample(n=500, random_state=42)
    return list(zip(sample["user_id"], sample["movie_id"], sample["rating"]))


# =============================================================================
# Known Test Cases Fixtures
# =============================================================================


@pytest.fixture
def known_user_movie_pairs() -> List[Dict[str, Any]]:
    """
    Known user-movie pairs from MovieLens 100K dataset.
    These are actual ratings that exist in the training data.
    """
    return [
        {"user_id": "196", "movie_id": "242", "actual_rating": 3.0},
        {"user_id": "186", "movie_id": "302", "actual_rating": 3.0},
        {"user_id": "22", "movie_id": "377", "actual_rating": 1.0},
        {"user_id": "244", "movie_id": "51", "actual_rating": 2.0},
        {"user_id": "166", "movie_id": "346", "actual_rating": 1.0},
    ]


@pytest.fixture
def unknown_users() -> List[str]:
    """User IDs that are unlikely to exist in the dataset."""
    return ["99999", "999999", "0", "-1", "new_user"]


@pytest.fixture
def unknown_movies() -> List[str]:
    """Movie IDs that are unlikely to exist in the dataset."""
    return ["99999", "999999", "0", "-1", "new_movie"]
