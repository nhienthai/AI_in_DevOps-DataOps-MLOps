"""
Airflow DAG for ML Training Pipeline.

This DAG orchestrates the movie rating prediction training pipeline:
1. Load Data
2. Preprocess Data
3. Train Model
4. Evaluate Model
5. Register Model (conditional)

Usage:
    Copy this file to your Airflow dags/ folder
    Access Airflow UI at http://localhost:8080
"""

from datetime import datetime, timedelta
import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
# EmptyOperator, not the old DummyOperator: airflow.operators.dummy was
# deprecated in Airflow 2.0 and removed in 2.9, so the old import breaks on
# any upgrade of the pinned 2.8.0 image.
from airflow.operators.empty import EmptyOperator

# =============================================================================
# Default Arguments
# =============================================================================
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# =============================================================================
# DAG Definition
# =============================================================================
# Weekly retraining. catchup=False so enabling the DAG does not backfill one
# run per week since the start_date.

dag = DAG(
    'movie_rating_training',
    default_args=default_args,
    description='ML Training Pipeline for Movie Rating Prediction',
    # `schedule`, not `schedule_interval`: the latter is deprecated since
    # Airflow 2.4 and removed in Airflow 3.
    schedule='@weekly',  # or '0 0 * * 0' for every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'movie-rating'],
)


# =============================================================================
# Task Functions
# =============================================================================

def load_data_task(**context):
    """
    Task 1: Load and prepare data.
    
    This function:
    1. Loads the dataset
    2. Splits into train/test
    3. Saves to temporary location
    4. Pushes metadata via XCom
    """
    from pipeline.data_ingestion import load_and_split
    
    print("Loading data...")
    trainset, testset, stats = load_and_split()
    
    # Isolate files by DAG run. A scheduled run can overlap a manually
    # triggered run; sharing one directory would let one run overwrite or
    # delete another run's pickles while they are still being consumed.
    dag_run = context.get('dag_run')
    run_id = getattr(dag_run, 'run_id', None) or context.get('run_id')
    if not run_id:
        raise ValueError("No Airflow run_id available for the working directory")

    safe_run_id = ''.join(
        char if char.isalnum() or char in '._-' else '_'
        for char in run_id
    )
    tmp_dir = os.path.join('/tmp/airflow_ml_pipeline', safe_run_id)
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open(f'{tmp_dir}/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(f'{tmp_dir}/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)
    
    # Push stats via XCom
    context['ti'].xcom_push(key='data_stats', value=stats)
    context['ti'].xcom_push(key='data_path', value=tmp_dir)
    
    print(f"Data loaded: {stats['n_ratings']} ratings")
    return "Data loaded successfully"


# =============================================================================
# Task 2: Preprocess and validate
# =============================================================================
def preprocess_data_task(**context):
    """
    Task 2: Preprocess and validate data.

    Loads the pickles written by load_data, runs the validation suite, and
    pushes the report to XCom.

    Raises:
        ValueError: Validation failed. Stopping here is deliberate — training on
            data that failed validation would produce a model nobody can trust,
            and the run would look successful.
    """
    from pipeline.preprocessing import preprocess_data

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if not tmp_dir:
        raise ValueError("No data_path in XCom — did load_data run?")

    with open(f'{tmp_dir}/trainset.pkl', 'rb') as f:
        trainset = pickle.load(f)
    with open(f'{tmp_dir}/testset.pkl', 'rb') as f:
        testset = pickle.load(f)

    report = preprocess_data(trainset, testset)

    if not report["preprocessing_successful"]:
        issues = (
            report["trainset_validation"]["issues"]
            + report["testset_validation"]["issues"]
        )
        raise ValueError(f"Data validation failed: {issues}")

    # XCom values are serialised, so push only the small scalar summary rather
    # than the whole nested report.
    context['ti'].xcom_push(key='preprocess_report', value={
        "successful": report["preprocessing_successful"],
        "n_users": report["trainset_validation"]["n_users"],
        "n_items": report["trainset_validation"]["n_items"],
        "n_ratings": report["trainset_validation"]["n_ratings"],
        "mean_rating": round(report["rating_distribution"]["mean"], 4),
    })

    print(f"Validation passed: {report['trainset_validation']['n_ratings']} train ratings")
    return "Preprocessing complete"


# =============================================================================
# Task 3: Train with MLflow tracking
# =============================================================================
# The weekly retrain uses the best configuration found by the hyperparameter
# sweep (see experiment_report.md), not the library defaults.
TRAINING_CONFIG = {
    "model_type": "svd",
    "n_factors": 50,
    "n_epochs": 20,
}


def train_model_task(**context):
    """
    Task 3: Train the model and log the run to MLflow.

    The run name carries the Airflow logical date, so every scheduled retrain is
    identifiable in the MLflow UI.
    """
    from pipeline.training import train_model, setup_mlflow

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if not tmp_dir:
        raise ValueError("No data_path in XCom — did load_data run?")

    with open(f'{tmp_dir}/trainset.pkl', 'rb') as f:
        trainset = pickle.load(f)

    setup_mlflow()

    params = {k: v for k, v in TRAINING_CONFIG.items() if k != "model_type"}
    model, run_id = train_model(
        trainset,
        model_type=TRAINING_CONFIG["model_type"],
        run_name=f"airflow_{context['ds']}",
        **params
    )

    with open(f'{tmp_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='run_id', value=run_id)

    print(f"Model trained. Run ID: {run_id}")
    return f"Model trained. Run ID: {run_id}"


# =============================================================================
# Task 4: Evaluate
# =============================================================================
def evaluate_model_task(**context):
    """
    Task 4: Evaluate the trained model and log metrics onto the training run.

    Metrics go to XCom as plain floats, which the branch task reads to decide
    whether this model is good enough to register.
    """
    from pipeline.evaluation import evaluate_model

    tmp_dir = context['ti'].xcom_pull(key='data_path')
    run_id = context['ti'].xcom_pull(key='run_id')

    if not run_id:
        raise ValueError("No run_id in XCom — did train_model run?")

    with open(f'{tmp_dir}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{tmp_dir}/testset.pkl', 'rb') as f:
        testset = pickle.load(f)

    metrics = evaluate_model(model, testset, run_id)

    # Keep XCom JSON-serialisable: drop None (mape can be None) and cast.
    context['ti'].xcom_push(key='metrics', value={
        k: float(v) for k, v in metrics.items() if v is not None
    })

    print(f"Evaluation complete. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")
    return f"Evaluation complete. RMSE: {metrics['rmse']:.4f}"


def decide_registration(**context):
    """
    Branch task: Decide whether to register model based on performance.
    
    Returns 'register_model' if RMSE < 1.0, otherwise 'skip_registration'
    """
    metrics = context['ti'].xcom_pull(key='metrics')
    
    if metrics and metrics.get('rmse', float('inf')) < 1.0:
        return 'register_model'
    return 'skip_registration'


def register_model_task(**context):
    """
    Task 5: Register the best model.
    """
    from pipeline.registry import register_best_model
    
    result = register_best_model()
    print(f"Model registered: {result['model_name']} v{result['version']}")
    return result


def cleanup_task(**context):
    """
    Final task: Cleanup temporary files.
    """
    import shutil
    
    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up: {tmp_dir}")
    
    return "Cleanup complete"


# =============================================================================
# Task Definitions
# =============================================================================

# Task 1: Load Data
t_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    dag=dag,
)

# Task 2: Preprocess Data
t_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)

# Task 3: Train Model
t_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Task 4: Evaluate Model
t_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Task 5: Branch - Decide Registration
t_decide = BranchPythonOperator(
    task_id='decide_registration',
    python_callable=decide_registration,
    dag=dag,
)

# Task 6a: Register Model
t_register = PythonOperator(
    task_id='register_model',
    python_callable=register_model_task,
    dag=dag,
)

# Task 6b: Skip Registration
t_skip = EmptyOperator(
    task_id='skip_registration',
    dag=dag,
)

# Task 7: Cleanup
t_cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_task,
    trigger_rule='none_failed',  # Run even if branch skipped
    dag=dag,
)


# =============================================================================
# Task Dependencies
# =============================================================================
# load_data -> preprocess -> train -> evaluate -> decide -> [register|skip] -> cleanup
t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide
t_decide >> [t_register, t_skip]
[t_register, t_skip] >> t_cleanup
