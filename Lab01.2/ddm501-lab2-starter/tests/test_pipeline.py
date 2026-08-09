"""
Unit tests for ML Pipeline.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Shared fixtures
# =============================================================================
@pytest.fixture(scope="session")
def split_data():
    """Load and split MovieLens once for the whole session — it is slow."""
    from pipeline.data_ingestion import load_and_split

    return load_and_split()


@pytest.fixture(scope="session")
def tiny_model(split_data):
    """A cheap trained model for tests that need predictions, not accuracy."""
    from surprise import SVD

    trainset, _, _ = split_data
    model = SVD(n_factors=5, n_epochs=2, random_state=42)
    model.fit(trainset)
    return model


@pytest.fixture(scope="session")
def tiny_predictions(tiny_model, split_data):
    """Predictions on a small slice of the test set."""
    _, testset, _ = split_data
    return tiny_model.test(testset[:500])


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_load_data_returns_dataset(self):
        """Test that load_data returns a dataset object."""
        from pipeline.data_ingestion import load_data
        
        data = load_data('ml-100k')
        assert data is not None
    
    def test_split_data_returns_train_test(self):
        """Test that split_data returns trainset and testset."""
        from pipeline.data_ingestion import load_data, split_data
        
        data = load_data('ml-100k')
        trainset, testset = split_data(data, test_size=0.2)
        
        assert trainset is not None
        assert testset is not None
        assert len(testset) > 0
    
    def test_get_data_stats(self):
        """Test that get_data_stats returns correct statistics."""
        from pipeline.data_ingestion import load_data, get_data_stats
        
        data = load_data('ml-100k')
        stats = get_data_stats(data)
        
        assert 'n_users' in stats
        assert 'n_items' in stats
        assert 'n_ratings' in stats
        assert stats['n_users'] > 0
        assert stats['n_items'] > 0


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_validate_trainset(self):
        """Test trainset validation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import validate_trainset
        
        trainset, _, _ = load_and_split()
        report = validate_trainset(trainset)
        
        assert 'is_valid' in report
        assert report['is_valid'] == True
    
    def test_get_rating_distribution(self):
        """Test rating distribution calculation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import get_rating_distribution
        
        trainset, _, _ = load_and_split()
        dist = get_rating_distribution(trainset)
        
        assert 'mean' in dist
        assert 'std' in dist
        assert 1.0 <= dist['mean'] <= 5.0


class TestTraining:
    """Tests for training module."""
    
    def test_list_available_models(self):
        """Test that available models are listed."""
        from pipeline.training import list_available_models
        
        models = list_available_models()
        assert 'svd' in models
        assert 'nmf' in models
        assert 'knn' in models
    
    def test_get_default_params(self):
        """Test getting default parameters."""
        from pipeline.training import get_default_params
        
        params = get_default_params('svd')
        assert 'n_factors' in params
        assert 'n_epochs' in params
    
    def test_get_model_class_returns_surprise_classes(self):
        """Each supported name maps to the right Surprise algorithm."""
        from surprise import SVD, NMF, KNNBasic
        from pipeline.training import get_model_class

        assert get_model_class('svd') is SVD
        assert get_model_class('nmf') is NMF
        assert get_model_class('knn') is KNNBasic

    def test_get_model_class_rejects_unknown(self):
        """An unsupported model type fails loudly, listing what is available."""
        from pipeline.training import get_model_class

        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_class('xgboost')

    def test_build_run_name_flattens_nested_params(self):
        """KNN's sim_options dict must not end up as a dict repr in the run name."""
        from pipeline.training import build_run_name

        name = build_run_name('knn', {
            'k': 40,
            'sim_options': {'name': 'cosine', 'user_based': True},
        })

        assert name == 'knn_k=40_name=cosine_user_based=True'
        assert '{' not in name

    def test_train_with_config_requires_model_type(self):
        """A config without model_type is a programming error, not a silent default."""
        from pipeline.training import train_with_config

        with pytest.raises(KeyError, match="model_type"):
            train_with_config(MagicMock(), {'n_factors': 50})

    def test_train_with_config_does_not_mutate_caller_config(self):
        """EXPERIMENT_CONFIGS entries are reused; popping from them would corrupt the sweep."""
        from pipeline.training import train_with_config

        config = {'model_type': 'svd', 'n_factors': 5, 'n_epochs': 1}
        original = dict(config)

        with patch('pipeline.training.train_model', return_value=('model', 'run')):
            train_with_config(MagicMock(), config)

        assert config == original

    def test_train_model_validates_before_starting_a_run(self):
        """A bad model type must not leave an empty run behind in MLflow."""
        from pipeline.training import train_model

        with patch('pipeline.training.mlflow.start_run') as start_run:
            with pytest.raises(ValueError):
                train_model(MagicMock(), model_type='nope')

        start_run.assert_not_called()

    @pytest.mark.slow
    def test_train_model_logs_params_and_returns_run_id(self, split_data):
        """End-to-end training against the live MLflow server."""
        from pipeline.training import train_model, setup_mlflow
        from mlflow.tracking import MlflowClient

        setup_mlflow()
        trainset, _, _ = split_data

        model, run_id = train_model(
            trainset, model_type='svd', run_name='pytest_train',
            n_factors=5, n_epochs=2,
        )

        assert model is not None
        assert run_id

        run = MlflowClient().get_run(run_id)
        assert run.data.params['model_type'] == 'svd'
        assert run.data.params['n_factors'] == '5'
        assert 'training_time_seconds' in run.data.metrics


class TestEvaluation:
    """Tests for evaluation module."""

    def test_create_prediction_distribution_plot(self, tiny_predictions):
        """Test plot creation."""
        from pipeline.evaluation import create_prediction_distribution_plot

        fig = create_prediction_distribution_plot(tiny_predictions)
        assert fig is not None

    def test_create_error_by_rating_plot(self, tiny_predictions):
        """The second plot renders too — it is logged as an artifact."""
        from pipeline.evaluation import create_error_by_rating_plot

        fig = create_error_by_rating_plot(tiny_predictions)
        assert fig is not None

    def test_additional_metrics_keys(self, tiny_predictions):
        """All documented metrics are present."""
        from pipeline.evaluation import calculate_additional_metrics

        metrics = calculate_additional_metrics(tiny_predictions)

        for key in ("mse", "rmse_manual", "mape", "coverage",
                    "n_predictions", "n_impossible"):
            assert key in metrics
        assert metrics["n_predictions"] == len(tiny_predictions)

    def test_rmse_manual_matches_surprise(self, tiny_predictions):
        """Our own RMSE agrees with Surprise's — a check on the metric maths."""
        from surprise import accuracy
        from pipeline.evaluation import calculate_additional_metrics

        expected = accuracy.rmse(tiny_predictions, verbose=False)
        actual = calculate_additional_metrics(tiny_predictions)["rmse_manual"]

        assert actual == pytest.approx(expected, abs=1e-9)

    def test_coverage_is_a_ratio(self, tiny_predictions):
        """Coverage is a fraction, and complements the impossible count."""
        from pipeline.evaluation import calculate_additional_metrics

        m = calculate_additional_metrics(tiny_predictions)

        assert 0.0 <= m["coverage"] <= 1.0
        assert m["coverage"] == pytest.approx(
            (m["n_predictions"] - m["n_impossible"]) / m["n_predictions"]
        )

    def test_additional_metrics_on_empty_input(self):
        """No predictions must not raise a ZeroDivisionError."""
        from pipeline.evaluation import calculate_additional_metrics

        m = calculate_additional_metrics([])

        assert m["n_predictions"] == 0
        assert m["coverage"] == 0.0

    def test_evaluate_model_rejects_empty_testset(self, tiny_model):
        """An empty test set is a caller error, not a silent 0.0 RMSE."""
        from pipeline.evaluation import evaluate_model

        with pytest.raises(ValueError, match="empty"):
            evaluate_model(tiny_model, [], run_id="x")

    def test_evaluate_model_without_mlflow(self, tiny_model, split_data):
        """Metrics can be computed without a tracking server."""
        from pipeline.evaluation import evaluate_model

        _, testset, _ = split_data
        metrics = evaluate_model(
            tiny_model, testset[:500], run_id="", log_to_mlflow=False
        )

        assert 0 < metrics["rmse"] < 5
        assert 0 < metrics["mae"] < 5


class TestRegistry:
    """Tests for registry module."""

    def test_list_registered_models(self):
        """Test listing registered models."""
        from pipeline.registry import list_registered_models

        # This should not raise an error
        models = list_registered_models()
        assert isinstance(models, list)

    def test_find_best_run_rejects_unknown_experiment(self):
        """A missing experiment is reported clearly, not as an IndexError."""
        from pipeline.registry import find_best_run

        with pytest.raises(ValueError, match="not found"):
            find_best_run(experiment_name="no-such-experiment-xyz")

    def test_transition_rejects_invalid_stage(self):
        """Only MLflow's four stages are accepted."""
        from pipeline.registry import transition_model_stage

        with pytest.raises(ValueError, match="Invalid stage"):
            transition_model_stage("any-model", "1", stage="Prodction")

    def test_register_model_rejects_run_without_artifact(self):
        """Registering a run that logged no model must fail at register time."""
        from pipeline.registry import register_model

        with patch('pipeline.registry.MlflowClient') as client_cls:
            client_cls.return_value.list_artifacts.return_value = []

            with pytest.raises(ValueError, match="no artifacts"):
                register_model("some-run-id", "some-model")

    @pytest.mark.slow
    def test_find_best_run_returns_lowest_rmse(self):
        """The best run really is the minimum, verified against all runs."""
        from mlflow.tracking import MlflowClient
        from pipeline.registry import find_best_run
        from pipeline.config import MLFLOW_EXPERIMENT_NAME

        best = find_best_run(MLFLOW_EXPERIMENT_NAME, metric="rmse")

        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        all_rmse = [
            r.data.metrics["rmse"]
            for r in client.search_runs([experiment.experiment_id])
            if "rmse" in r.data.metrics
        ]

        assert best["metrics"]["rmse"] == pytest.approx(min(all_rmse))


class TestConfig:
    """Tests for configuration."""
    
    def test_config_values(self):
        """Test that config has required values."""
        from pipeline.config import (
            DATASET_NAME,
            TEST_SIZE,
            DEFAULT_MODEL_TYPE,
            MODEL_CONFIGS,
        )
        
        assert DATASET_NAME == 'ml-100k'
        assert 0 < TEST_SIZE < 1
        assert DEFAULT_MODEL_TYPE in ['svd', 'nmf', 'knn']
        assert 'svd' in MODEL_CONFIGS


# =============================================================================
# Experiment runner
# =============================================================================
class TestExperimentRunner:
    """Tests for the hyperparameter sweep and its report."""

    def test_report_handles_failed_experiments(self, tmp_path):
        """A failed config appears in the report instead of crashing it."""
        from experiments.run_experiments import generate_experiment_report

        results = [
            {
                "config": {"model_type": "svd", "n_factors": 50},
                "run_id": "abc12345def",
                "run_name": "svd_n_factors=50",
                "metrics": {"rmse": 0.93, "mae": 0.73, "mse": 0.86,
                            "training_time_seconds": 0.2},
            },
            {"config": {"model_type": "nmf", "n_factors": 10}, "error": "boom"},
        ]

        report = generate_experiment_report(results, str(tmp_path / "r.md"))

        assert "Successful: **1**" in report
        assert "Failed: **1**" in report
        assert "boom" in report

    def test_report_ranks_by_rmse(self, tmp_path):
        """The results table is ordered best-first."""
        from experiments.run_experiments import generate_experiment_report

        results = [
            {"config": {"model_type": "nmf"}, "run_id": "b" * 12,
             "run_name": "worse", "metrics": {"rmse": 1.1, "mae": 0.9, "mse": 1.2,
                                              "training_time_seconds": 1.0}},
            {"config": {"model_type": "svd"}, "run_id": "a" * 12,
             "run_name": "better", "metrics": {"rmse": 0.9, "mae": 0.7, "mse": 0.8,
                                               "training_time_seconds": 0.1}},
        ]

        report = generate_experiment_report(results, str(tmp_path / "r.md"))

        assert report.index("| 1 | SVD") < report.index("| 2 | NMF")
        assert "Best RMSE: **0.9000**" in report

    def test_report_survives_all_failures(self, tmp_path):
        """A sweep where nothing worked still produces a readable report."""
        from experiments.run_experiments import generate_experiment_report

        results = [{"config": {"model_type": "svd"}, "error": "no server"}]

        report = generate_experiment_report(results, str(tmp_path / "r.md"))

        assert "Successful: **0**" in report

    def test_all_experiment_configs_are_valid(self):
        """Every config in EXPERIMENT_CONFIGS names a supported model."""
        from pipeline.config import EXPERIMENT_CONFIGS
        from pipeline.training import get_model_class

        assert len(EXPERIMENT_CONFIGS) >= 5, "the lab requires at least 5 experiments"
        for config in EXPERIMENT_CONFIGS:
            assert "model_type" in config
            get_model_class(config["model_type"])  # raises if unsupported


# =============================================================================
# Airflow DAG
# =============================================================================
class TestDag:
    """The DAG file is a deliverable; check it without needing Airflow installed."""

    def test_dag_file_defines_expected_tasks(self):
        """All seven tasks and the dependency chain are present."""
        from pathlib import Path

        source = Path(__file__).resolve().parent.parent / "dags" / "ml_training_dag.py"
        text = source.read_text()

        for task_id in ("load_data", "preprocess_data", "train_model",
                        "evaluate_model", "decide_registration",
                        "register_model", "skip_registration", "cleanup"):
            assert f"'{task_id}'" in text, f"task {task_id} missing from the DAG"

        assert "t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide" in text
        assert "schedule_interval='@weekly'" in text

    def test_dag_has_no_unimplemented_stubs(self):
        """No task function was left as a bare `pass`."""
        from pathlib import Path

        source = Path(__file__).resolve().parent.parent / "dags" / "ml_training_dag.py"
        lines = [ln.strip() for ln in source.read_text().splitlines()]

        assert "pass  # Remove this and implement" not in lines


# =============================================================================
# Integration Tests
# =============================================================================
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Run every stage against the live MLflow server."""
        from pipeline.run_pipeline import run_pipeline

        results = run_pipeline(model_type="svd", n_factors=5, n_epochs=2)

        assert results["status"] == "completed"
        assert results["stages"]["training"]["run_id"]
        assert 0 < results["stages"]["evaluation"]["metrics"]["rmse"] < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
