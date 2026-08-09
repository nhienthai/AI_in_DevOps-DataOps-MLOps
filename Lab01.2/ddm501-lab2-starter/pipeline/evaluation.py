"""
Model Evaluation Stage for ML Pipeline.

This module handles:
- Making predictions on test data
- Calculating evaluation metrics
- Logging metrics to MLflow
- Creating evaluation visualizations
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib

# Agg has no GUI event loop. Airflow workers and pytest run without a display,
# and the default macOS backend would abort there.
matplotlib.use("Agg")

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from surprise import accuracy

from pipeline.config import ARTIFACTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(
    model: Any,
    testset: List,
    run_id: str,
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model on the test set and log metrics to its MLflow run.

    Metrics land on the *training* run (reopened by run_id) rather than a new
    run, so the MLflow UI shows parameters and metrics on one row.

    Args:
        model: Trained Surprise model
        testset: Test set as list of (user, item, rating) tuples
        run_id: MLflow run ID to log metrics to
        log_to_mlflow: Whether to log metrics to MLflow

    Returns:
        Dictionary with evaluation metrics: rmse, mae, mse, mape, coverage,
        n_predictions, n_impossible.

    Raises:
        ValueError: testset is empty — there is nothing to evaluate.

    Example:
        metrics = evaluate_model(model, testset, run_id)
        print(f"RMSE: {metrics['rmse']:.4f}")
    """
    if not testset:
        raise ValueError("testset is empty; nothing to evaluate")

    logger.info(f"Evaluating model on {len(testset)} test ratings...")
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    metrics = {"rmse": float(rmse), "mae": float(mae)}
    metrics.update(calculate_additional_metrics(predictions))

    if log_to_mlflow:
        if not run_id:
            raise ValueError("run_id is required when log_to_mlflow is True")

        # nested=False + an explicit run_id reopens the finished training run.
        with mlflow.start_run(run_id=run_id):
            for name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(name, value)

            fig = create_prediction_distribution_plot(predictions)
            mlflow.log_figure(fig, "plots/prediction_distribution.png")
            plt.close(fig)

            fig = create_error_by_rating_plot(predictions)
            mlflow.log_figure(fig, "plots/error_by_rating.png")
            plt.close(fig)

            mlflow.set_tag("stage", "evaluated")

    logger.info(f"Evaluation complete. RMSE={rmse:.4f}, MAE={mae:.4f}")
    return metrics


# =============================================================================
# Additional metrics
# =============================================================================
def calculate_additional_metrics(predictions: List) -> Dict[str, float]:
    """
    Calculate evaluation metrics beyond RMSE and MAE.

    Metrics returned:
        mse           Mean squared error.
        rmse_manual   RMSE recomputed from raw errors — a cross-check on
                      Surprise's own accuracy.rmse.
        mape          Mean absolute percentage error, in percent. MovieLens
                      ratings are >= 1 so no division by zero occurs in
                      practice, but zero actuals are excluded anyway.
        coverage      Share of test pairs the model could actually estimate.
                      Surprise flags cold-start pairs as `was_impossible` and
                      falls back to the global mean; a low coverage means the
                      RMSE above is mostly measuring that fallback.
        n_predictions Number of predictions scored.
        n_impossible  Count of cold-start fallbacks.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Dictionary with additional metrics
    """
    if not predictions:
        return {
            "mse": 0.0,
            "rmse_manual": 0.0,
            "mape": None,
            "coverage": 0.0,
            "n_predictions": 0,
            "n_impossible": 0,
        }

    actuals = np.array([pred.r_ui for pred in predictions], dtype=float)
    estimated = np.array([pred.est for pred in predictions], dtype=float)

    errors = estimated - actuals
    mse = float(np.mean(errors ** 2))

    non_zero = actuals != 0
    mape = (
        float(np.mean(np.abs(errors[non_zero] / actuals[non_zero])) * 100)
        if np.any(non_zero)
        else None
    )

    n_impossible = sum(
        1 for pred in predictions if pred.details.get("was_impossible", False)
    )
    coverage = float((len(predictions) - n_impossible) / len(predictions))

    return {
        "mse": mse,
        "rmse_manual": float(np.sqrt(mse)),
        "mape": mape,
        "coverage": coverage,
        "n_predictions": len(predictions),
        "n_impossible": n_impossible,
    }


# =============================================================================
# Visualization Functions (PROVIDED)
# =============================================================================
def create_prediction_distribution_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing prediction vs actual rating distribution.
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Matplotlib figure
    """
    actuals = [pred.r_ui for pred in predictions]
    estimated = [pred.est for pred in predictions]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scatter plot of actual vs predicted
    axes[0].scatter(actuals, estimated, alpha=0.1, s=1)
    axes[0].plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Rating')
    axes[0].set_ylabel('Predicted Rating')
    axes[0].set_title('Actual vs Predicted Ratings')
    axes[0].legend()
    
    # Plot 2: Distribution of actual ratings
    axes[1].hist(actuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Actual Ratings')
    
    # Plot 3: Distribution of prediction errors
    errors = np.array(estimated) - np.array(actuals)
    axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Prediction Errors')
    
    plt.tight_layout()
    return fig


def create_error_by_rating_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing error distribution by actual rating.
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Matplotlib figure
    """
    # Group predictions by actual rating
    rating_groups = {}
    for pred in predictions:
        rating = round(pred.r_ui)
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(pred.est - pred.r_ui)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratings = sorted(rating_groups.keys())
    positions = range(len(ratings))
    
    bp = ax.boxplot(
        [rating_groups[r] for r in ratings],
        positions=positions,
        widths=0.6
    )
    
    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error by Actual Rating')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    return fig


def save_evaluation_report(metrics: Dict, filepath: str) -> None:
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value}\n")
    
    logger.info(f"Evaluation report saved to {filepath}")


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Evaluation Module")
    print("=" * 50)
    
    # Evaluate a quick model without touching MLflow.
    from surprise import SVD

    from pipeline.data_ingestion import load_and_split

    trainset, testset, _ = load_and_split()
    model = SVD(n_factors=10, n_epochs=5, random_state=42)
    model.fit(trainset)

    metrics = evaluate_model(model, testset, run_id="", log_to_mlflow=False)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name:22} {value}")
