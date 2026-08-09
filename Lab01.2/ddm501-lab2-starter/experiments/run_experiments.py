"""
Experiment Runner - Run multiple experiments for hyperparameter tuning.

This script runs multiple experiments with different configurations
and logs all results to MLflow for comparison.

Usage:
    python -m experiments.run_experiments
"""

import logging
from typing import Dict, Any, List
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import (
    EXPERIMENT_CONFIGS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from pipeline.data_ingestion import load_and_split
from pipeline.training import build_run_name, train_model, setup_mlflow
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs

# All experiments land in this MLflow experiment, kept separate from the
# single-model pipeline runs so the comparison view stays clean.
TUNING_EXPERIMENT_NAME = "hyperparameter-tuning"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Single experiment
# =============================================================================
def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = TUNING_EXPERIMENT_NAME
) -> Dict[str, Any]:
    """
    Train and evaluate one configuration, logging both to a single MLflow run.

    Args:
        trainset: Training data
        testset: Test data
        config: Configuration dictionary with model_type and hyperparameters
        experiment_name: Name of the MLflow experiment

    Returns:
        Dictionary with keys config, run_id, run_name, metrics.

    Raises:
        KeyError: config has no 'model_type'.
        ValueError: model_type is unknown.
    """
    mlflow.set_experiment(experiment_name)

    config_copy = dict(config)
    model_type = config_copy.pop("model_type")
    run_name = build_run_name(model_type, config_copy)

    model, run_id = train_model(
        trainset,
        model_type=model_type,
        run_name=run_name,
        **config_copy
    )

    evaluate_model(model, testset, run_id)

    # Read metrics back from the run instead of using evaluate_model's return
    # value: the run also carries training_time_seconds, logged during training,
    # and this guarantees the report shows exactly what MLflow stored.
    metrics = MlflowClient().get_run(run_id).data.metrics

    return {
        "config": config,
        "run_id": run_id,
        "run_name": run_name,
        "metrics": metrics,
    }


# =============================================================================
# All experiments
# =============================================================================
def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = TUNING_EXPERIMENT_NAME
) -> List[Dict[str, Any]]:
    """
    Run every configuration in `configs`.

    Data is loaded and split once so all runs are scored on the *same* test set
    — otherwise the RMSE values would not be comparable across runs, which is
    the whole point of the sweep.

    A failing config is recorded and the sweep continues: one bad
    hyperparameter combination should not discard the runs that already worked.

    Args:
        configs: List of configuration dictionaries
        experiment_name: Name of the MLflow experiment

    Returns:
        List of results; failed entries carry an 'error' key instead of 'metrics'.
    """
    logger.info(f"Running {len(configs)} experiments...")

    trainset, testset, stats = load_and_split()
    logger.info(
        f"Shared split: {trainset.n_ratings} train / {len(testset)} test ratings"
    )

    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"\n[{i}/{len(configs)}] {config}")
        try:
            result = run_single_experiment(trainset, testset, config, experiment_name)
            results.append(result)
            logger.info(
                f"  RMSE={result['metrics']['rmse']:.4f} "
                f"MAE={result['metrics']['mae']:.4f}"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"config": config, "error": str(e)})

    n_ok = sum(1 for r in results if "metrics" in r)
    logger.info(f"\nFinished: {n_ok}/{len(configs)} experiments succeeded")
    return results


# =============================================================================
# Reporting
# =============================================================================
def _format_params(config: Dict[str, Any]) -> str:
    """Render a config's hyperparameters as a compact, table-safe string."""
    parts = []
    for key, value in config.items():
        if key == "model_type":
            continue
        if isinstance(value, dict):
            parts.extend(f"{k}={v}" for k, v in value.items())
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "defaults"


def _analyse(successful: List[Dict[str, Any]]) -> List[str]:
    """Derive the observations section from the results themselves."""
    lines = []

    # --- Which algorithm family wins ----------------------------------------
    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    for r in successful:
        by_algo.setdefault(r["config"]["model_type"], []).append(r)

    lines.append("### Algorithm comparison\n")
    lines.append("| Algorithm | Runs | Best RMSE | Worst RMSE | Mean RMSE |")
    lines.append("|-----------|-----:|----------:|-----------:|----------:|")
    for algo in sorted(by_algo, key=lambda a: min(r["metrics"]["rmse"] for r in by_algo[a])):
        rmses = [r["metrics"]["rmse"] for r in by_algo[algo]]
        lines.append(
            f"| {algo.upper()} | {len(rmses)} | {min(rmses):.4f} | "
            f"{max(rmses):.4f} | {sum(rmses) / len(rmses):.4f} |"
        )
    lines.append("")

    ranked = sorted(by_algo, key=lambda a: min(r["metrics"]["rmse"] for r in by_algo[a]))
    if len(ranked) > 1:
        best_algo, worst_algo = ranked[0], ranked[-1]
        best_v = min(r["metrics"]["rmse"] for r in by_algo[best_algo])
        worst_v = min(r["metrics"]["rmse"] for r in by_algo[worst_algo])
        lines.append(
            f"{best_algo.upper()} beats {worst_algo.upper()} by "
            f"{worst_v - best_v:.4f} RMSE ({(worst_v - best_v) / worst_v * 100:.1f}%) "
            f"at each family's best setting.\n"
        )

    # --- Does capacity help? -------------------------------------------------
    svd = [r for r in successful if r["config"]["model_type"] == "svd"]
    factor_groups: Dict[int, List[float]] = {}
    for r in svd:
        n_factors = r["config"].get("n_factors")
        if n_factors is not None:
            factor_groups.setdefault(n_factors, []).append(r["metrics"]["rmse"])

    if len(factor_groups) > 1:
        lines.append("### Effect of n_factors (SVD)\n")
        lines.append("| n_factors | Best RMSE |")
        lines.append("|----------:|----------:|")
        for n_factors in sorted(factor_groups):
            lines.append(f"| {n_factors} | {min(factor_groups[n_factors]):.4f} |")
        lines.append("")

        ordered = sorted(factor_groups)
        first, last = min(factor_groups[ordered[0]]), min(factor_groups[ordered[-1]])
        direction = "improves" if last < first else "degrades"
        lines.append(
            f"Going from {ordered[0]} to {ordered[-1]} factors {direction} RMSE by "
            f"{abs(last - first):.4f}. More capacity is not automatically better "
            f"here — MovieLens 100K is small enough that large factor counts "
            f"overfit unless regularisation rises with them.\n"
        )

    # --- Cost vs benefit -----------------------------------------------------
    timed = [r for r in successful if r["metrics"].get("training_time_seconds")]
    if timed:
        fastest = min(timed, key=lambda r: r["metrics"]["training_time_seconds"])
        slowest = max(timed, key=lambda r: r["metrics"]["training_time_seconds"])
        best = min(successful, key=lambda r: r["metrics"]["rmse"])
        lines.append("### Training cost\n")
        lines.append(
            f"- Fastest: `{fastest['run_name']}` at "
            f"{fastest['metrics']['training_time_seconds']:.2f}s "
            f"(RMSE {fastest['metrics']['rmse']:.4f})"
        )
        lines.append(
            f"- Slowest: `{slowest['run_name']}` at "
            f"{slowest['metrics']['training_time_seconds']:.2f}s "
            f"(RMSE {slowest['metrics']['rmse']:.4f})"
        )
        lines.append(
            f"- Best RMSE: `{best['run_name']}` at "
            f"{best['metrics'].get('training_time_seconds', float('nan')):.2f}s\n"
        )
        spread = slowest["metrics"]["rmse"] - fastest["metrics"]["rmse"]
        if abs(spread) < 0.01:
            lines.append(
                f"The slowest run buys only {abs(spread):.4f} RMSE over the fastest "
                f"— not worth "
                f"{slowest['metrics']['training_time_seconds'] / max(fastest['metrics']['training_time_seconds'], 1e-9):.1f}x "
                f"the training time for weekly retraining.\n"
            )

    # --- Cold start ----------------------------------------------------------
    coverages = {r["metrics"].get("coverage") for r in successful}
    coverages.discard(None)
    if coverages and max(coverages) < 1.0:
        lines.append(
            f"Coverage tops out at {max(coverages):.3f}: some test pairs are "
            f"cold-start and fall back to the global mean, which flatters RMSE.\n"
        )
    elif coverages:
        lines.append(
            "Coverage is 1.000 across all runs — every test pair was scored by "
            "the model itself, so no RMSE is inflated by global-mean fallbacks.\n"
        )

    return lines


def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a markdown comparison report from experiment results.

    Args:
        results: List of experiment results from run_all_experiments()
        output_path: Path to save the report

    Returns:
        Report content as string
    """
    successful = [r for r in results if "metrics" in r]
    failed = [r for r in results if "metrics" not in r]

    report: List[str] = []
    report.append("# Experiment Report — Movie Rating Prediction")
    report.append(f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    report.append(
        "All runs share one train/test split (80/20, `random_state=42`) of "
        "MovieLens 100K, so the RMSE values below are directly comparable.\n"
    )

    # --- Summary -------------------------------------------------------------
    report.append("## Summary\n")
    report.append(f"- Total experiments: **{len(results)}**")
    report.append(f"- Successful: **{len(successful)}**")
    report.append(f"- Failed: **{len(failed)}**")

    if successful:
        best = min(successful, key=lambda r: r["metrics"]["rmse"])
        report.append(
            f"- Best RMSE: **{best['metrics']['rmse']:.4f}** "
            f"(`{best['run_name']}`)"
        )
    report.append("")

    # --- Results table -------------------------------------------------------
    report.append("## All results\n")
    report.append("Sorted by RMSE, lower is better.\n")
    report.append("| # | Model | Parameters | RMSE | MAE | MSE | Train (s) | Run ID |")
    report.append("|--:|-------|------------|-----:|----:|----:|----------:|--------|")

    for i, r in enumerate(sorted(successful, key=lambda r: r["metrics"]["rmse"]), 1):
        m = r["metrics"]
        report.append(
            f"| {i} | {r['config']['model_type'].upper()} "
            f"| {_format_params(r['config'])} "
            f"| {m['rmse']:.4f} | {m['mae']:.4f} | {m['mse']:.4f} "
            f"| {m.get('training_time_seconds', float('nan')):.2f} "
            f"| `{r['run_id'][:8]}` |"
        )
    report.append("")

    if failed:
        report.append("### Failed experiments\n")
        report.append("| Model | Parameters | Error |")
        report.append("|-------|------------|-------|")
        for r in failed:
            report.append(
                f"| {r['config'].get('model_type', '?')} "
                f"| {_format_params(r['config'])} | {r.get('error', 'unknown')} |"
            )
        report.append("")

    # --- Best model ----------------------------------------------------------
    if successful:
        best = min(successful, key=lambda r: r["metrics"]["rmse"])
        report.append("## Best model\n")
        report.append(f"- Run name: `{best['run_name']}`")
        report.append(f"- Run ID: `{best['run_id']}`")
        report.append(f"- Configuration: `{json.dumps(best['config'], sort_keys=True)}`")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|------:|")
        for name in ("rmse", "mae", "mse", "mape", "coverage", "training_time_seconds"):
            value = best["metrics"].get(name)
            if value is not None:
                report.append(f"| {name} | {value:.4f} |")
        report.append("")

        # --- Analysis --------------------------------------------------------
        report.append("## Analysis\n")
        report.extend(_analyse(successful))

        # --- Recommendation --------------------------------------------------
        report.append("## Recommendation\n")
        params = {k: v for k, v in best["config"].items() if k != "model_type"}
        report.append(
            f"Deploy **{best['config']['model_type'].upper()}** with "
            f"`{json.dumps(params, sort_keys=True)}`. It has the lowest RMSE "
            f"({best['metrics']['rmse']:.4f}) on the shared test split and trains in "
            f"{best['metrics'].get('training_time_seconds', float('nan')):.2f}s, "
            f"which fits comfortably inside a weekly retraining window.\n"
        )
        report.append(
            "The Airflow DAG registers this configuration automatically: it "
            "promotes the best run to the `Production` stage only when RMSE is "
            "below 1.0, so a degraded retrain cannot silently replace a good "
            "model.\n"
        )

    content = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Report written to {output_path}")
    return content


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Run all experiments and generate report."""
    
    logger.info("=" * 60)
    logger.info("Starting Experiment Runner")
    logger.info("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Run experiments
    results = run_all_experiments(
        configs=EXPERIMENT_CONFIGS,
        experiment_name=TUNING_EXPERIMENT_NAME
    )
    
    # Generate report
    report = generate_experiment_report(results, "experiment_report.md")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Summary")
    logger.info("=" * 60)
    
    successful = [r for r in results if 'metrics' in r]
    if successful:
        best = min(successful, key=lambda x: x['metrics'].get('rmse', float('inf')))
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Best RMSE: {best['metrics']['rmse']:.4f}")
        logger.info(f"Best config: {best['config']}")
    
    # Compare top runs
    # Read the top runs back from MLflow rather than from the in-memory
    # results, so this also proves the runs really landed on the server.
    logger.info("\nTop 5 runs (queried from MLflow):")
    top_runs = compare_runs(
        experiment_name=TUNING_EXPERIMENT_NAME, metric="rmse", top_n=5
    )
    for i, run in enumerate(top_runs, 1):
        logger.info(
            f"  {i}. RMSE={run['metrics'].get('rmse', float('nan')):.4f} - "
            f"{run['params'].get('model_type')} ({run['run_id'][:8]})"
        )

    logger.info("\nReport saved to: experiment_report.md")
    logger.info(f"View experiments in MLflow UI: {MLFLOW_TRACKING_URI}")
    
    return results


if __name__ == "__main__":
    main()
