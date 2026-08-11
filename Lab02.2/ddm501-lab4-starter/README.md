# Lab 4: Monitoring & Production Deployment

A production-style movie rating API with Prometheus metrics, Grafana dashboards, alert rules, and a load-testing script. This lab extends the movie-rating model from the earlier labs and adds the observability layer used in a real deployment.

## What is included

- FastAPI API with `/predict`, `/predict/batch`, `/health`, `/metrics`, and `/model/info`
- Prometheus metrics for HTTP traffic, model health, prediction latency, and prediction errors
- Grafana dashboards for system metrics and ML metrics
- Prometheus alert rules for API health and model health
- A load-testing script to generate traffic and populate the dashboards

## Project structure

```text
ddm501-lab4-starter/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── metrics.py
│   ├── middleware.py
│   ├── config.py
│   └── schemas.py
├── prometheus/
│   ├── prometheus.yml
│   └── alerts/
│       ├── api_alerts.yml
│       └── ml_alerts.yml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── dashboards/
│   │       └── dashboards.yml
│   └── dashboards/
│       ├── system_dashboard.json
│       └── ml_dashboard.json
├── scripts/
│   ├── train_model.py
│   └── load_test.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.13 for the local Windows venv, or Docker for the containerized stack
- Docker and Docker Compose
- About 4 GB free disk space for the monitoring containers
- A trained model file at `models/svd_model.pkl`

## Local setup

Create a virtual environment and install the local dependencies:

```bash
cd ddm501-lab4-starter
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The local pin set is chosen to install cleanly on Windows Python 3.13. The Docker image uses Python 3.10-slim and the same application code.

## Model handoff from earlier labs

This lab does not train a new model by itself. It loads the movie-rating model
from `models/svd_model.pkl`, so the file must come from a previous training run
in Lab01.1 or from `python scripts/train_model.py` in this lab.

The practical workflow is:

1. Train the model in Lab01.1 or Lab02.2.
2. Copy the resulting `svd_model.pkl` into this lab's `models/` folder.
3. Start the API from this lab's root directory so `app.main` can be imported.

If the model file is missing, train it first:

```bash
python scripts/train_model.py
```

## Run the API

Start the API locally:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `http://localhost:8000/health` (returns HTTP 503 until the model is loaded)
- `http://localhost:8000/metrics`
- `http://localhost:8000/docs`

## Run the monitoring stack

Start the full stack with Prometheus and Grafana:

```bash
docker compose up -d --build
```

Services:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana uses the default credentials `admin` / `admin`.

Stop everything with:

```bash
docker compose down
```

## Load testing

Run a basic load test against the API:

```bash
python scripts/load_test.py --duration 60 --workers 10
```

The script aborts unless `/health` reports both `status=healthy` and
`model_loaded=true`. Point it at another deployment with `--url` or the
`API_URL` environment variable:

```bash
python scripts/load_test.py --url https://api.example.com --duration 60
```

Batch mode:

```bash
python scripts/load_test.py --duration 60 --workers 10 --batch
```

Spike test:

```bash
python scripts/load_test.py --spike --workers 10
```

## Metrics and dashboards

The API exports these metrics:

- `http_requests_total`
- `http_request_duration_seconds`
- `ml_predictions_total`
- `ml_prediction_duration_seconds`
- `ml_prediction_value`
- `ml_prediction_errors_total`
- `ml_model_loaded`
- `ml_model_info`
- `ml_model_last_reload_timestamp`
- `ml_batch_prediction_size`

Prometheus scrapes the API every 10 seconds and loads alert rules from `prometheus/alerts`.

Grafana auto-loads the dashboards from `grafana/dashboards`.

## Recommended workflow

1. Create the venv and install dependencies.
2. Train the model if `models/svd_model.pkl` is missing.
3. Start `docker compose up -d --build`.
4. Open Grafana and confirm the dashboards are populated.
5. Run the load test to generate traffic and verify the charts.

## Troubleshooting

- If the API returns `503`, check that `models/svd_model.pkl` exists.
- If you see `ModuleNotFoundError: No module named 'app'`, run the command
	from `Lab02.2/ddm501-lab4-starter` or use `python -m uvicorn app.main:app --app-dir .`.
- If the Error Rate panel shows `No data`, that usually means there were no 5xx
	responses in the selected time range. Generate some requests, or trigger an
	error and refresh the panel.
- If Prometheus shows no targets, confirm the API container is running and that `prometheus.yml` points to `api:8000`.
- If Grafana shows empty dashboards, verify that the Prometheus datasource has loaded and that the stack was started with `docker compose up -d --build`.
- If local installation fails on Python 3.13, recreate the venv with the bundled requirements in this folder.
