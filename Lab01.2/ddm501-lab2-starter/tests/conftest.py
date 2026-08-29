"""Shared pytest configuration.

The suite must not need a running MLflow server, and must not care what anyone
else has logged. Two things broke that before:

* The default tracking URI is ``http://localhost:5001``. That server reports an
  artifact root of ``/mlflow/artifacts`` — a path that exists inside the MLflow
  container and nowhere else — so a client running on the host tries to create
  it and dies with ``OSError: [Errno 30] Read-only file system: '/mlflow'``.
  Three tests failed that way on any machine that was not the container.
* One test asserted against whatever runs happened to already be in the shared
  experiment, so whether it passed depended on who had run what beforehand.

Every test session therefore gets its own file-backed tracking store in a
temporary directory, created *before* ``pipeline.config`` is imported (that
module reads ``os.getenv`` at import time, so setting the variable later would
be too late — the module-level constant would already hold the old value).

To run the same tests against a real MLflow server instead — useful as a
smoke test of the deployed stack, not as part of CI — set:

    LAB2_LIVE_MLFLOW=1 pytest
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# --- Isolation, applied at import time --------------------------------------
# conftest.py is imported before any test module, and the test modules import
# `pipeline` lazily inside their functions, so the environment is already set
# by the time pipeline.config reads it.

_USING_LIVE_SERVER = bool(os.getenv("LAB2_LIVE_MLFLOW"))
_TEMP_STORE: Path | None = None

if not _USING_LIVE_SERVER:
    _TEMP_STORE = Path(tempfile.mkdtemp(prefix="lab2-mlruns-"))
    os.environ["MLFLOW_TRACKING_URI"] = _TEMP_STORE.as_uri()
    # A separate experiment name as well, so a stray local ./mlruns from a
    # manual pipeline run can never be mistaken for test output.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "movie-rating-prediction-tests"


def _active_tracking_uri() -> str:
    """The tracking URI in force, whether set here or left to pipeline.config.

    In live mode nothing is exported, so the value comes from the package default
    rather than the environment — reading os.environ directly would raise.
    """
    from pipeline.config import MLFLOW_TRACKING_URI

    return os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI


@pytest.fixture(scope="session", autouse=True)
def mlflow_store() -> str:
    """Yield the tracking URI in use, and clean up the temporary store after.

    Yields:
        The MLflow tracking URI the whole session is pointed at.
    """
    yield _active_tracking_uri()
    if _TEMP_STORE is not None:
        shutil.rmtree(_TEMP_STORE, ignore_errors=True)


@pytest.fixture(scope="session")
def experiment_name() -> str:
    """The experiment name the isolated session logs into."""
    from pipeline.config import MLFLOW_EXPERIMENT_NAME

    return MLFLOW_EXPERIMENT_NAME


def pytest_report_header(config) -> str:
    """Show which tracking store the run used, so a failure is diagnosable."""
    mode = "LIVE SERVER" if _USING_LIVE_SERVER else "isolated temp store"
    return f"mlflow: {mode} -> {_active_tracking_uri()}"
