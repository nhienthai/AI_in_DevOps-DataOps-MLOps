# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 1: First ML Product
# =============================================================================

FROM python:3.10-slim

# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# System packages
# -----------------------------------------------------------------------------
# - build-essential: scikit-surprise ships as an sdist and compiles its Cython
#   extension at install time.
# - curl: used by the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependencies (copied first so Docker caches this layer)
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY . .

# Do not run the API as root.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/svd_model.pkl

# -----------------------------------------------------------------------------
# Port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------------------------------
# Startup command
# -----------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
