"""
Configuration settings for the Movie Rating API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "svd_model.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# API settings
API_TITLE = "Movie Rating Prediction API"
API_DESCRIPTION = "API for predicting movie ratings using collaborative filtering"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Cache settings (Redis)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))
CACHE_TIMEOUT_SECONDS = float(os.getenv("CACHE_TIMEOUT_SECONDS", 0.2))

# Metrics settings
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
