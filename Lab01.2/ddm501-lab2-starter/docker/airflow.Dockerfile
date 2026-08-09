# =============================================================================
# Airflow image with the ML pipeline's dependencies
# DDM501 - Lab 2
#
# The stock apache/airflow image has neither MLflow nor scikit-surprise, so DAG
# tasks that import pipeline.* would fail at runtime. This image adds them.
# =============================================================================

# The plain apache/airflow:2.8.0 tag is built on Python 3.8, where
# matplotlib 3.8.x is unavailable (it needs >= 3.9). The -python3.11 variant is
# the same Airflow release on a newer interpreter.
FROM apache/airflow:2.8.0-python3.11

# scikit-surprise ships as an sdist and compiles a Cython extension on install.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# mlflow-skinny, not mlflow: the workers only need the tracking/registry client.
# The full package drags in Flask, Alembic and Gunicorn pins that fight with
# Airflow's own versions of the same libraries.
RUN pip install --no-cache-dir \
      "numpy<2" \
      "scikit-surprise==1.1.4" \
      "mlflow-skinny==2.9.2" \
      "matplotlib==3.8.2"

# Bake the dataset into the image. Surprise prompts on stdin the first time a
# built-in dataset is missing, and an Airflow task has no stdin to answer with.
RUN python -c "from surprise import Dataset; Dataset.load_builtin('ml-100k', prompt=False)"

USER root
RUN apt-get purge -y --auto-remove build-essential || true
USER airflow
