"""
Model Registry Stage for ML Pipeline.

This module handles:
- Finding the best model from experiments
- Registering models to MLflow Model Registry
- Managing model versions and stages
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pipeline.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ARTIFACT_PATH,
    MLFLOW_REGISTERED_MODEL_NAME,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Finding the best run
# =============================================================================
def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True
) -> Dict[str, Any]:
    """
    Find the best run in an experiment, ranked by a metric.

    Runs that never logged the metric (a crashed training, or a run that was
    trained but never evaluated) are excluded by the filter string. Without it
    MLflow sorts missing values first and "best" would be a run with no score.

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (default: 'rmse')
        ascending: If True, lower is better (default: True for RMSE)

    Returns:
        Dictionary with keys run_id, metrics, params, artifact_uri, run_name.

    Raises:
        ValueError: The experiment does not exist, or has no run with the metric.

    Example:
        best = find_best_run(metric='rmse', ascending=True)
        print(f"Best RMSE: {best['metrics']['rmse']}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            "Run the pipeline or the experiment runner first."
        )

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > -1e30",  # excludes runs missing the metric
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No run in experiment '{experiment_name}' has a '{metric}' metric."
        )

    best_run = runs[0]
    logger.info(
        f"Best run {best_run.info.run_id}: "
        f"{metric}={best_run.data.metrics.get(metric)}"
    )

    return {
        "run_id": best_run.info.run_id,
        "run_name": best_run.data.tags.get("mlflow.runName", ""),
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifact_uri": best_run.info.artifact_uri,
    }


# =============================================================================
# Registering a model
# =============================================================================
VALID_STAGES = ("None", "Staging", "Production", "Archived")


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = MLFLOW_MODEL_ARTIFACT_PATH
) -> str:
    """
    Register a model from an MLflow run into the Model Registry.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path to the model artifact within the run

    Returns:
        Version number of the registered model (as string)

    Raises:
        ValueError: The run has no artifact at `artifact_path`, so registering
            would create an empty version that fails at load time.

    Example:
        version = register_model(run_id, "movie-rating-model")
        print(f"Registered model version: {version}")
    """
    client = MlflowClient()

    # Registering a non-existent path succeeds silently and only breaks later,
    # when something tries to load the model. Fail here instead.
    artifacts = [f.path for f in client.list_artifacts(run_id, artifact_path)]
    if not artifacts:
        raise ValueError(
            f"Run {run_id} has no artifacts under '{artifact_path}'. "
            "Was the model logged during training?"
        )

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Registering model from {model_uri} as '{model_name}'")

    result = mlflow.register_model(model_uri, model_name)

    logger.info(f"Model registered: {model_name} version {result.version}")
    return str(result.version)


# =============================================================================
# Stage transitions
# =============================================================================
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production",
    archive_existing: bool = True,
) -> None:
    """
    Move a model version to a new stage.

    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('None', 'Staging', 'Production', 'Archived')
        archive_existing: Archive whatever currently occupies the stage, so
            exactly one version is ever in Production.

    Raises:
        ValueError: `stage` is not one of the four MLflow stages.

    Example:
        transition_model_stage("movie-rating-model", "1", "Production")
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage {stage!r}. Valid stages: {list(VALID_STAGES)}")

    client = MlflowClient()
    logger.info(f"Transitioning {model_name} v{version} to {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing and stage in ("Staging", "Production"),
    )

    logger.info(f"Model {model_name} v{version} is now in {stage}")


# =============================================================================
# Find + register + promote
# =============================================================================
def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    metric: str = "rmse",
    stage: str = "Production",
    ascending: bool = True,
) -> Dict[str, Any]:
    """
    Find the best run in an experiment and promote its model.

    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name for the registered model
        metric: Metric to optimize (default: 'rmse')
        stage: Stage to transition to (default: 'Production')
        ascending: True when a lower metric is better (RMSE, MAE)

    Returns:
        Dictionary with run_id, model_name, version, stage, metric, params.

    Raises:
        ValueError: No suitable run exists, or the run has no model artifact.

    Example:
        result = register_best_model()
        print(f"Registered {result['model_name']} v{result['version']}")
    """
    best_run = find_best_run(experiment_name, metric, ascending=ascending)
    logger.info(
        f"Best run: {best_run['run_id']} "
        f"with {metric}={best_run['metrics'].get(metric)}"
    )

    version = register_model(best_run["run_id"], model_name)
    transition_model_stage(model_name, version, stage)

    return {
        "run_id": best_run["run_id"],
        "run_name": best_run["run_name"],
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "metric": metric,
        "metrics": best_run["metrics"],
        "params": best_run["params"],
    }


def load_registered_model(
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    stage: str = "Production",
) -> Any:
    """
    Download and unpickle the model currently in a given stage.

    Closes the loop on the registry: proves the registered artifact is a usable
    Surprise model, not just a database row.

    Args:
        model_name: Name of the registered model
        stage: Stage to load from

    Returns:
        The unpickled Surprise model.

    Raises:
        ValueError: Nothing is registered in that stage.
    """
    import pickle
    from pathlib import Path

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ValueError(f"No version of '{model_name}' is in stage '{stage}'")

    version = versions[0]
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path=MLFLOW_MODEL_ARTIFACT_PATH
    )

    pickles = sorted(Path(local_dir).glob("*.pkl"))
    if not pickles:
        raise ValueError(f"No .pkl artifact found for {model_name} v{version.version}")

    with open(pickles[0], "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded {model_name} v{version.version} ({stage}) from {pickles[0].name}")
    return model


# =============================================================================
# Helper Functions (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models.
    
    Returns:
        List of model information dictionaries
    """
    client = MlflowClient()
    models = client.search_registered_models()
    
    return [
        {
            "name": model.name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in model.latest_versions
            ]
        }
        for model in models
    ]


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the current production version of a model.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Dictionary with model info or None if not found
    """
    client = MlflowClient()
    
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return {
                "name": model_name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
            }
    except Exception as e:
        logger.error(f"Error getting production model: {e}")
    
    return None


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top N runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of runs to return
        
    Returns:
        List of run information sorted by metric
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n
    )
    
    return [
        {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Registry Module")
    print("=" * 50)
    
    import mlflow
    from pipeline.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("\nRegistered models:")
    for model in list_registered_models():
        print(f"  {model['name']}")
        for v in model["latest_versions"]:
            print(f"    v{v['version']:>3}  {v['stage']}")

    production = get_production_model(MLFLOW_REGISTERED_MODEL_NAME)
    print(f"\nCurrent production model: {production}")

    print("\nTo promote the best run of an experiment:")
    print("  python -c 'from pipeline.registry import register_best_model;"
          " print(register_best_model())'")
