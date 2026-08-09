# =============================================================================
# Dockerfile for Movie Rating Prediction API with Monitoring
# DDM501 - Lab 4: Monitoring & Production Deployment
# =============================================================================

FROM python:3.10-slim AS builder

WORKDIR /build

# scikit-surprise is distributed as source and needs a compiler to build its
# wheel. Keep the toolchain out of the final runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache optimization)
COPY requirements.txt .

# Install the pre-built wheels without carrying a compiler into production.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
