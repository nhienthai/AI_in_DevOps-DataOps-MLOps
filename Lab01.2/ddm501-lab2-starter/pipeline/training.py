"""
Model Training Stage for ML Pipeline.

This module handles:
- Model initialization
- Model training
- MLflow experiment tracking
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import mlflow
from surprise import SVD, NMF, KNNBasic

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ARTIFACT_PATH,
    MODEL_CONFIGS,
    MODELS_DIR,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Classes Registry
# =============================================================================
MODEL_CLASSES = {
    "svd": SVD,
    "nmf": NMF,
    "knn": KNNBasic,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")


# =============================================================================
# Training
# =============================================================================
def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params
) -> Tuple[Any, str]:
    """
    Train a recommendation model and log everything to MLflow.

    The run is left OPEN-then-closed here; evaluation reopens it by run_id so
    that parameters and metrics land on the same run.

    Args:
        trainset: Surprise trainset object
        model_type: Type of model ('svd', 'nmf', 'knn')
        run_name: Optional name for the MLflow run
        **model_params: Model hyperparameters

    Returns:
        Tuple of (trained_model, run_id)

    Raises:
        ValueError: model_type is not one of MODEL_CLASSES.
        TypeError: a hyperparameter is not accepted by the model class.

    Example:
        model, run_id = train_model(
            trainset,
            model_type='svd',
            n_factors=100,
            n_epochs=20
        )
    """
    # Validate before opening a run, so a typo does not leave an empty run
    # sitting in the experiment.
    model_class = get_model_class(model_type)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        mlflow.log_param("model_type", model_type)
        for key, value in model_params.items():
            # KNN's sim_options is a dict; MLflow params must be scalars, so
            # flatten it into sim_options.name / sim_options.user_based.
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}.{sub_key}", sub_value)
                mlflow.log_param(key, json.dumps(value, sort_keys=True))
            else:
                mlflow.log_param(key, value)

        mlflow.log_param("n_train_users", trainset.n_users)
        mlflow.log_param("n_train_items", trainset.n_items)
        mlflow.log_param("n_train_ratings", trainset.n_ratings)

        mlflow.set_tags({
            "stage": "training",
            "algorithm": model_type.upper(),
            "framework": "surprise",
        })

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        logger.info(f"Training {model_type} model with {model_params}...")
        model = model_class(**model_params)

        started = time.perf_counter()
        model.fit(trainset)
        training_seconds = time.perf_counter() - started

        # Training time belongs to the run: it is what makes two configs with
        # the same RMSE distinguishable.
        mlflow.log_metric("training_time_seconds", training_seconds)

        # ---------------------------------------------------------------------
        # Artifacts
        # ---------------------------------------------------------------------
        # A per-run filename keeps concurrent experiments from overwriting one
        # another's pickle before it is uploaded.
        model_path = MODELS_DIR / f"model_{model_type}_{run_id[:8]}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(model_path), artifact_path=MLFLOW_MODEL_ARTIFACT_PATH)

        logger.info(
            f"Training complete in {training_seconds:.2f}s. Run ID: {run_id}"
        )

    return model, run_id


# =============================================================================
# Config-driven training
# =============================================================================
def train_with_config(
    trainset: Any,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
) -> Tuple[Any, str]:
    """
    Train a model from a configuration dictionary.

    Args:
        trainset: Surprise trainset object
        config: Configuration dictionary with model_type and hyperparameters
        run_name: Optional name for the MLflow run

    Returns:
        Tuple of (trained_model, run_id)

    Raises:
        KeyError: config has no 'model_type' key.

    Example:
        config = {"model_type": "svd", "n_factors": 100, "n_epochs": 20}
        model, run_id = train_with_config(trainset, config)
    """
    if "model_type" not in config:
        raise KeyError(f"config must contain 'model_type'. Got keys: {list(config)}")

    # Copy so the caller's config (often a shared list entry) is untouched.
    config_copy = dict(config)
    model_type = config_copy.pop("model_type")

    if run_name is None:
        run_name = build_run_name(model_type, config_copy)

    return train_model(trainset, model_type=model_type, run_name=run_name, **config_copy)


def build_run_name(model_type: str, params: Dict[str, Any]) -> str:
    """
    Build a readable run name so the MLflow UI is scannable.

    Nested params (KNN's sim_options) are flattened to `name=value` rather than
    dumped as a dict, which would make the run list unreadable.

    Example:
        >>> build_run_name("svd", {"n_factors": 100, "n_epochs": 20})
        'svd_n_factors=100_n_epochs=20'
    """
    parts = []
    for key, value in params.items():
        if isinstance(value, dict):
            parts.extend(f"{sub_key}={sub_value}" for sub_key, sub_value in value.items())
        else:
            parts.append(f"{key}={value}")
    return "_".join([model_type, *parts])


# =============================================================================
# Model class lookup
# =============================================================================
def get_model_class(model_type: str):
    """
    Get the Surprise model class for a given model type.

    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')

    Returns:
        Model class from the Surprise library

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Supported types: {list(MODEL_CLASSES.keys())}"
        )
    return MODEL_CLASSES[model_type]


# =============================================================================
# Helper functions (PROVIDED)
# =============================================================================
def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary of default parameters
    """
    return MODEL_CONFIGS.get(model_type, {})


def list_available_models() -> list:
    """
    List all available model types.
    
    Returns:
        List of model type names
    """
    return list(MODEL_CLASSES.keys())


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split
    
    print("Testing Training Module")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    trainset, testset, _ = load_and_split()
    
    # Test training (uncomment after implementing)
    # model, run_id = train_model(
    #     trainset,
    #     model_type="svd",
    #     run_name="test_run",
    #     n_factors=50,
    #     n_epochs=10
    # )
    # print(f"Model trained. Run ID: {run_id}")
    
    print("\nAvailable models:", list_available_models())
    print("Default SVD params:", get_default_params("svd"))
