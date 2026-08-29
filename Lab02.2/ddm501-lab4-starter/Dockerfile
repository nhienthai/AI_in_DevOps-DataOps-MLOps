# =============================================================================
# Dockerfile for Movie Rating Prediction API with Monitoring
# DDM501 - Lab 4: Monitoring & Production Deployment
# =============================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - build-essential: scikit-surprise ships as an sdist and compiles its Cython
#   extension at install time, so gcc must exist at pip-install time.
# - curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for cache optimization)
COPY requirements.txt .

# Install Python dependencies, then drop the toolchain again so the runtime
# image does not ship a compiler it will never use.
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

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
