# Lab 1: First ML Product — Movie Rating Prediction API

A production-shaped ML product: an SVD collaborative-filtering model trained on
MovieLens 100K, wrapped in a FastAPI REST service, containerized with Docker,
and covered by a pytest suite.

**Course:** DDM501 — Lab 1 (First ML Product)

---

## Features

- **SVD collaborative filtering** trained on MovieLens 100K (100,000 ratings, 943 users, 1,682 movies)
- **REST API** — `/predict`, `/predict/batch`, `/health`, `/model/info`
- **Input validation** via Pydantic v2 (non-empty, length-bounded string IDs)
- **Graceful cold start** — unknown users/movies fall back to the global mean instead of erroring
- **Docker + Compose** with a container `HEALTHCHECK` and a non-root runtime user
- **Redis prediction cache** that fails open — Redis down means slower, never broken
- **Prometheus metrics** at `/metrics`, scraped by a Prometheus service in the same stack
- **56 pytest tests** covering happy paths, validation errors, edge cases, caching, metrics, and the OpenAPI contract
- **Auto-generated Swagger docs** at `/docs`

### Model performance

5-fold cross-validation on MovieLens 100K:

| Metric | Value |
|--------|-------|
| RMSE   | 0.9353 |
| MAE    | 0.7374 |

Hyperparameters: `n_factors=100`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`.

---

## Project Structure

```
ddm501-lab1-starter/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application + endpoints
│   ├── model.py             # MovieRatingModel wrapper (load / predict / batch)
│   ├── schemas.py           # Pydantic request/response models
│   ├── cache.py             # Optional Redis prediction cache
│   ├── metrics.py           # Prometheus metric definitions
│   └── config.py            # Configuration (env-overridable)
├── models/
│   ├── svd_model.pkl        # Trained model (generated, git-ignored)
│   └── model_metadata.json  # Training metrics, served by /model/info
├── tests/
│   ├── __init__.py
│   └── test_api.py          # 40 unit tests
├── scripts/
│   └── train_model.py       # Training script
├── Dockerfile
├── docker-compose.yml       # api + redis + prometheus
├── prometheus.yml           # Prometheus scrape config
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.10+** (verified on 3.10 in Docker and 3.12 locally)
- **Docker & Docker Compose** (for the containerized path)
- **Git**
- A C compiler is needed on first install — `scikit-surprise` builds a Cython
  extension from source. macOS: Xcode command line tools. Debian/Ubuntu:
  `build-essential`.

---

## Quick Start

### 1. Setup

```bash
cd ddm501-lab1-starter

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

> **Note on pinned versions:** `numpy` is pinned below 2.0 because
> `scikit-surprise` compiles against the numpy 1.x C ABI. `scikit-surprise` is
> pinned to `1.1.4` rather than `1.1.3` — the older release has no PEP 517 build
> metadata and fails to build on Python 3.11+.

### 2. Train the Model

```bash
python scripts/train_model.py
```

Downloads MovieLens 100K (first run only), runs 5-fold cross-validation, trains
on the full dataset, and writes `models/svd_model.pkl` plus
`models/model_metadata.json`. Takes about 30 seconds.

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: <http://localhost:8000/docs>

### 4. Run with Docker

```bash
docker compose build
docker compose up -d

docker compose ps          # all three services should show (healthy)
docker compose logs -f api
docker compose down
```

The stack brings up three services:

| Service | Port | Role |
|---------|------|------|
| `api` | 8000 | The prediction API |
| `redis` | 6379 | Prediction cache |
| `prometheus` | 9090 | Scrapes `api:8000/metrics` every 15s |

The image does not contain the model file — `docker-compose.yml` mounts
`./models` into `/app/models` at runtime, so retraining does not require a
rebuild. Train the model **before** starting the container.

---

## Caching and Monitoring

### Redis cache

`/predict` and `/predict/batch` read through a Redis cache keyed by
`pred:{model_version}:{user_id}:{movie_id}` with a 1-hour TTL. The model version
is part of the key, so retraining under a new version cannot serve stale
ratings. Redis runs with `maxmemory 128mb` and `allkeys-lru`: cache entries are
disposable, so the coldest keys are evicted rather than letting Redis refuse
writes.

**The cache fails open.** Every operation is wrapped so that a Redis outage
costs speed, never availability:

```bash
docker compose stop redis
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"user_id":"186","movie_id":"302"}'
# still 200 OK — served straight from the model
```

After a failure the cache is skipped for 30 seconds, so a dead Redis does not
add a connection timeout to every request. It reconnects on its own once Redis
comes back — no API restart needed.

The cache is **off by default** (`CACHE_ENABLED=false`) so `uvicorn app.main:app`
works with no extra service; `docker-compose.yml` switches it on.

Inspect it directly:

```bash
docker exec ddm501-redis redis-cli KEYS 'pred:*'
docker exec ddm501-redis redis-cli TTL 'pred:1.0.0:196:242'
```

### Prometheus metrics

`GET /metrics` serves the Prometheus text format. Prometheus UI:
<http://localhost:9090> — check **Status → Targets**, `movie-rating-api` should
be `UP`.

The expression browser starts empty by design; paste a query from the list below
into the **Expression** box and press Execute.

| Metric | Type | Labels |
|--------|------|--------|
| `http_requests_total` | counter | `method`, `path`, `status` |
| `http_request_duration_seconds` | histogram | `method`, `path` |
| `predictions_total` | counter | `endpoint` |
| `prediction_cache_events_total` | counter | `result` (`hit`/`miss`) |
| `model_loaded` | gauge | — |
| `cache_connected` | gauge | — |
| `model_info` | gauge | `version`, `type` |
| `model_training_metric` | gauge | `metric` (`rmse`/`mae`) |

Latency labels use the **route template** (`/predict`), never the raw URL, so
cardinality stays bounded. `/metrics` excludes itself from the request counters.

Useful queries:

Paste **one** expression at a time into the Prometheus expression box — it
evaluates a single query, so a pasted block of several is a parse error.

```promql
sum by (path) (rate(http_requests_total[1m]))
```
```promql
sum(rate(http_requests_total{status=~"4..|5.."}[1m])) / sum(rate(http_requests_total[1m]))
```
```promql
histogram_quantile(0.95, sum by (le, path) (rate(http_request_duration_seconds_bucket[1m])))
```
```promql
sum(rate(prediction_cache_events_total{result="hit"}[1m])) / sum(rate(prediction_cache_events_total[1m]))
```
```promql
model_training_metric{metric="rmse"}
```

> The `sum()` around the cache-hit ratio is required, not cosmetic. Without it
> both sides still carry the `result` label, so Prometheus matches `hit` against
> `hit` and the ratio is always exactly `1`.

Rate windows must be at least 4× the 15s scrape interval — `[1m]` is the
practical minimum here; `[10s]` returns nothing.

> **Note:** metrics live in the process's default registry, which assumes a
> single worker. Running uvicorn with `--workers N` would need
> `prometheus_client`'s multiprocess mode.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | API metadata |
| GET  | `/health` | Health check — reports whether the model is loaded |
| GET  | `/model/info` | Model version, type, training metrics, cache state |
| POST | `/predict` | Predict a rating for one user-movie pair |
| POST | `/predict/batch` | Predict up to 100 pairs in one request |
| GET  | `/metrics` | Prometheus metrics |
| GET  | `/docs` | Swagger UI |
| GET  | `/openapi.json` | OpenAPI schema |

Every response carries an `X-Process-Time-Ms` header with server-side latency.

### Usage Examples

**Health check**

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy", "model_loaded": true}
```

**Single prediction**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "196", "movie_id": "242"}'
```

```json
{
  "user_id": "196",
  "movie_id": "242",
  "predicted_rating": 3.8,
  "model_version": "1.0.0"
}
```

**Batch prediction**

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"predictions": [
        {"user_id": "196", "movie_id": "242"},
        {"user_id": "186", "movie_id": "302"}
      ]}'
```

```json
{
  "predictions": [
    {"user_id": "196", "movie_id": "242", "predicted_rating": 3.8, "model_version": "1.0.0"},
    {"user_id": "186", "movie_id": "302", "predicted_rating": 3.71, "model_version": "1.0.0"}
  ],
  "total_count": 2
}
```

**Model info**

```bash
curl http://localhost:8000/model/info
```

### Error Handling

| Status | When | `detail` shape |
|--------|------|----------------|
| `422 Unprocessable Entity` | Missing field, blank ID, ID longer than 64 chars, malformed JSON, batch over 100 items | Array of per-field errors (FastAPI `HTTPValidationError`) |
| `503 Service Unavailable` | Model file missing or failed to load | String |
| `500 Internal Server Error` | Unexpected failure during prediction | String |

All validation failures share the one `HTTPValidationError` shape — blank IDs are
rejected by a Pydantic field validator rather than by the model wrapper, so
clients never have to handle two different `detail` types. Surrounding
whitespace in an ID is trimmed, not rejected: `" 196 "` scores as `"196"`.

Unknown user or movie IDs are **not** errors. Surprise falls back to the global
mean rating, so cold-start requests return `200` with a valid 1.0–5.0 rating.
Predictions are always clipped to the 1–5 MovieLens scale and rounded to two
decimals.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" -d '{"movie_id": "242"}'
# 422 — user_id is required
```

---

## Configuration

All settings are environment-overridable (see [`app/config.py`](app/config.py)):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/svd_model.pkl` | Path to the pickled model |
| `MODEL_VERSION` | `1.0.0` | Version string returned in predictions, and part of the cache key |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `DEBUG` | `false` | Debug flag |
| `CACHE_ENABLED` | `false` | Turn the Redis cache on (compose sets `true`) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL_SECONDS` | `3600` | Lifetime of a cached prediction |
| `CACHE_TIMEOUT_SECONDS` | `0.2` | Socket timeout for cache I/O |
| `METRICS_ENABLED` | `true` | Record Prometheus metrics |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html
```

The suite requires a trained model — run `python scripts/train_model.py` first.

**Coverage:**

- Health and root endpoints (including the uvicorn lifespan path)
- `/predict` happy path, response schema, determinism, ID echoing
- Validation errors: missing fields, empty body, wrong types, malformed JSON, `405` on GET
- Edge cases: unknown user, unknown movie, both unknown, SQL/XSS-shaped IDs, empty/blank/overlong IDs, whitespace trimming, rounding
- `/model/info` version, type, load state, and training metrics
- `/predict/batch` ordering, agreement with `/predict`, empty list, over-limit rejection
- `MovieRatingModel` unit tests: missing file raises, batch/single consistency, known vs. cold-start IDs
- OpenAPI contract: every endpoint declares a response schema and a description, and the documented `422` matches the body actually returned
- Metrics: exposition format, expected series present, counter increments, route-template labels
- Cache: disabled by default, unreachable Redis does not raise, predictions still served, key includes model version

The suite runs **without Redis** — that is deliberate, since it proves the cache
is optional rather than a hidden dependency.

---

## Architecture Notes

**Model loading.** `app/main.py` loads the model once at import time and again in
the FastAPI `startup` event. Import-time loading means a `TestClient` created
without the lifespan context still gets a ready model; the startup hook covers
the uvicorn path. A load failure is logged, not raised — the process stays up and
`/health` reports `unhealthy` so an orchestrator can restart or route away.

**Why string IDs.** Surprise's raw IDs are the strings from the source dataset.
Passing an integer would look up a different key and silently return the global
mean, so the schema takes strings and the wrapper coerces with `str()`.

**Batch limit.** `/predict/batch` accepts at most 100 pairs to keep tail latency
bounded; larger batches are rejected with `422`.

---

## Deliverables Checklist

- [x] `app/model.py` — `_load_model()`, `predict()`, `predict_batch()`
- [x] `app/schemas.py` — request/response models with examples
- [x] `app/main.py` — `/predict` endpoint with error handling
- [x] `Dockerfile` — with health check
- [x] `docker-compose.yml` — ports, volumes, environment, restart policy, health check
- [x] `tests/test_api.py` — happy path and edge case tests
- [x] Bonus: additional compose services — **Redis** cache (wired into `/predict`) and **Prometheus** (scraping a real `/metrics` endpoint)
- [x] Bonus: `/predict/batch`, `/model/info` metrics, `X-Process-Time-Ms` header, non-root container user
