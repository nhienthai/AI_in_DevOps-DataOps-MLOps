"""
Configuration settings for the Movie Rating API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "svd_model.pkl"))
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_TYPE: str = "SVD (Collaborative Filtering)"

# API settings
API_TITLE: str = "Movie Rating Prediction API"
API_DESCRIPTION: str = "API for predicting movie ratings using collaborative filtering"
API_VERSION: str = "1.0.0"

# Server settings
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# Rating constraints
MIN_RATING: float = 1.0
MAX_RATING: float = 5.0

# Identifier constraints, shared by every request schema so the contract cannot
# drift between /predict and /predict/batch.
MAX_ID_LENGTH: int = 64

# Cache settings (Redis) - carried over from Lab 1. Off by default; the compose
# stack and CI turn it on.
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_TIMEOUT_SECONDS: float = float(os.getenv("CACHE_TIMEOUT_SECONDS", "0.2"))

# Metrics settings
METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
