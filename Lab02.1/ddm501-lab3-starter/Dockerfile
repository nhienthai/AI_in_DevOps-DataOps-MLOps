# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 3: Testing & CI/CD
#
# Multi-stage: the builder compiles wheels (scikit-surprise is a source
# distribution and needs gcc), the runtime image installs them and carries no
# toolchain. Smaller image, smaller attack surface.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: build wheels
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/svd_model.pkl

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Application code. `training/` is copied because a model produced by the
# NumPy backend references training.local_svd at unpickle time.
COPY app/ ./app/
COPY training/ ./training/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check uses the interpreter that is already present - no curl in the
# image, so nothing extra to install or patch.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
