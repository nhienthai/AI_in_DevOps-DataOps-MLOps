# Lab 3: Testing & CI/CD for ML Systems

[![Lab3 CI](https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps/actions/workflows/lab3-ci.yml/badge.svg)](https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps/actions/workflows/lab3-ci.yml)
![tests](https://img.shields.io/badge/tests-278%20passed-brightgreen)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)

Comprehensive testing and CI/CD for the movie rating prediction API carried
forward from Labs 1 and 2 — including its Redis cache and Prometheus metrics.

| | |
| --- | --- |
| **Tests** | 278 (90 unit · 109 integration · 48 data · 31 behavioural) |
| **Coverage** | **100 %** of `app/` in CI (required: ≥ 80 %) |
| **Runtime** | ~2.4 s for the full suite in CI |
| **Model** | SVD collaborative filtering, RMSE 0.944 / MAE 0.745 |
| **Quality** | black · isort · flake8 · mypy — all clean |

> Locally the suite reports **273 passed, 5 skipped and 99.5 %**: the five
> skipped tests need a live Redis. CI runs them against a Redis service
> container, which is what closes the last two lines of `app/cache.py`. Both
> numbers are real — see [`docs/screenshots/`](docs/screenshots/) for the CI run.

---

## Project Structure

```
ddm501-lab3-starter/
├── app/                         # Serving layer (100 % covered in CI)
│   ├── main.py                  # FastAPI application
│   ├── model.py                 # ML model wrapper + cold-start detection
│   ├── schemas.py               # Pydantic schemas
│   ├── cache.py                 # Redis prediction cache (fails open)
│   ├── metrics.py               # Prometheus metrics
│   └── config.py                # Configuration
├── training/                    # Build-time only, not shipped logic
│   ├── dataset.py               # MovieLens 100K download + cache
│   ├── local_svd.py             # NumPy-only SVD (fallback backend)
│   └── validators.py            # Custom data validators
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # 90 tests — model wrapper, schemas
│   ├── integration/             # 109 tests — API (54), cache (30), metrics (25)
│   ├── data/                    # 48 tests — data quality & validators
│   └── model/                   # 31 tests — behavioural
├── .github/workflows/
│   ├── ci.yml                   # CI (standalone-repo version)
│   └── cd.yml                   # CD — tag → Docker Hub/GHCR → staging → prod
├── scripts/
│   ├── train_model.py           # Train + evaluate + persist
│   └── validate_model.py        # Model quality gate for CI
├── docs/
│   ├── TESTING_STRATEGY.md      # Testing strategy document
│   └── screenshots/             # Passing-CI evidence, one note per image
├── models/                      # svd_model.pkl, metrics.json, metadata
├── .pre-commit-config.yaml      # Hooks, version-pinned to match CI
├── .flake8                      # flake8 config (not readable from pyproject)
├── pyproject.toml               # black / isort / mypy / pytest / coverage
├── Dockerfile                   # Multi-stage, non-root, python healthcheck
├── docker-compose.yml           # api + redis + prometheus (from Lab 1)
├── prometheus.yml               # Scrape config
└── requirements*.txt
```

> **Note on the CI badge:** GitHub Actions only runs workflows in `.github/workflows/`
> at the **repository root**. Because this lab lives inside a monorepo, the
> workflow that actually executes is `/.github/workflows/lab3-ci.yml` at the repo
> root; `.github/workflows/ci.yml` here is the same pipeline for when this folder
> is pushed as a standalone repository. Keep the two in sync.

---

## Quick Start

### 1. Setup

```bash
cd ddm501-lab3-starter

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Train the model

```bash
python scripts/train_model.py                    # auto-picks a backend
python scripts/train_model.py --backend local --cv 3   # no compiler needed
```

This downloads MovieLens 100K (~5 MB, cached in `~/.surprise_data`), runs
cross-validation, and writes `models/svd_model.pkl`, `models/metrics.json` and
`models/model_metadata.json`.

<details>
<summary><strong>Two training backends — why</strong></summary>

`scikit-surprise` ships **source-only** on PyPI (no wheels for any platform), so
installing it needs a C toolchain. That is fine on Linux CI and in Docker, but it
fails on Windows without MSVC Build Tools — which would make the entire test
suite unrunnable on a typical dev laptop.

`scripts/train_model.py` therefore supports two interchangeable backends that
produce the same artifact contract (`model.predict(uid, iid).est`):

| Backend | Implementation | Requirement | Quality |
| --- | --- | --- | --- |
| `surprise` | `surprise.SVD` | C compiler | RMSE 0.935 |
| `local` | `training.local_svd.LocalSVD` (NumPy) | none | RMSE 0.944 |

`--backend auto` (the default) picks `surprise` when it imports and falls back to
`local` otherwise, so the same command works everywhere. The serving layer never
imports either one — it just unpickles an object with a `.predict()` method.

</details>

### 3. Run the tests

```bash
pytest tests/ -v                                   # all 278
pytest tests/ -v --cov=app --cov-report=html       # + coverage → htmlcov/
pytest tests/ -m "not slow"                        # skip the throughput test

pytest tests/unit/ -v            # 90   — model wrapper, schemas
pytest tests/integration/ -v     # 109  — API, cache, metrics
pytest tests/data/ -v            # 48   — data quality & validators
pytest tests/model/ -v           # 31   — invariance, directional, robustness
```

### 4. Code quality

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
pre-commit run --all-files

black app/ tests/ scripts/ training/
isort app/ tests/ scripts/ training/
flake8 app/ tests/ scripts/ training/
mypy app/ training/ scripts/ --ignore-missing-imports
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 8000
# docs: http://localhost:8000/docs
```

```bash
docker build -t movie-rating-api:local .
docker run -p 8000:8000 movie-rating-api:local
```

Full stack with the cache and monitoring from Lab 1:

```bash
python scripts/train_model.py       # the image is model-free; models/ is mounted
docker compose up --build
# API        http://localhost:8000/docs
# Prometheus http://localhost:9090
```

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | API information |
| `GET` | `/health` | Health + `model_loaded` + `cache_connected` |
| `GET` | `/model/info` | Version, type, load state, training metrics |
| `GET` | `/metrics` | Prometheus exposition format |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Up to 100 predictions |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"196","movie_id":"242"}'
```

```json
{
  "user_id": "196",
  "movie_id": "242",
  "predicted_rating": 3.97,
  "model_version": "1.0.0"
}
```

**Status codes:** `200` ok · `422` validation error · `404` unknown route ·
`405` wrong method · `503` model not loaded · `500` inference failure.

`503` rather than `500` for a missing model is deliberate: it tells a load
balancer to route elsewhere.

---

## Test Suite

Full rationale in **[docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md)**.

### Unit — `tests/unit/` (90)

Model wrapper in isolation (load failure, corrupted pickle, a pickle with no
`predict()`, return type, rounding, range clipping via a stub algorithm, batch
order, blank-ID rejection, cold-start detection, singleton) and the Pydantic
contract (required fields, `MAX_ID_LENGTH` boundaries, whitespace handling,
batch bounds 0/1/100/101, rating boundaries 1.0/5.0, the typed info schemas).

### Integration — `tests/integration/` (109)

- **`test_api.py` (54)** — real HTTP through `TestClient`: every endpoint, the
  OpenAPI document, CORS, and every error path. `503` when the model is missing
  or reports itself unloaded mid-request, `422` when the model rejects an ID,
  `500` when inference raises — produced by monkeypatching `app.main.model`.
- **`test_cache.py` (30)** — the Redis cache's defining property: it **fails
  open**. Unreachable Redis, read failures, write failures, corrupted entries
  and the backoff all degrade to model-only serving. Plus the part that matters
  most — that a cache *hit* actually skips the model, so a silently dead cache
  cannot pass unnoticed.
- **`test_metrics.py` (25)** — the exposition format, the content type
  Prometheus dispatches on, every series a dashboard needs, that counters
  actually move (including on 422s), that labels use the route *template* (an
  unbounded-cardinality bug otherwise), and the `X-Process-Time-Ms` header.

### Data — `tests/data/` (48)

Custom validators (`training/validators.py`) tested against clean **and**
deliberately corrupted data, then the real MovieLens 100K validated against its
contract: 100,000 rows, 943 users, 1,682 movies, no nulls, whole-star ratings in
`[1,5]`, no duplicate pairs, mean rating within `[3.4, 3.7]`, ≥ 20 ratings per
user.

### Behavioural — `tests/model/` (31)

- **Invariance** — determinism, batch order/size independence, pickle round-trip
- **Directional** — *Star Wars* > *Cape Fear* for the same user; a generous user
  scores above a harsh one; cold start lands near the global mean
- **Minimum functionality** — predictions are not all identical, spread > 0.3
- **Robustness** — unknown IDs, 200-char IDs, zero-padded IDs, injection strings
- **Performance** — MAE, fraction within 1.5, worst error, recorded RMSE

---

## Continuity with Lab 1

The Lab 3 starter shipped a simplified `app/`, which would have made this lab a
*regression* on the service built in Lab 1. The serving layer was realigned so
the system under test is the system that shipped:

| Restored from Lab 1 | Why it matters here |
| --- | --- |
| `ModelNotLoadedError` | Lets the API answer `503` (retry elsewhere) instead of `500` |
| Blank-ID rejection in `predict()` | Caller error surfaces as `422`, not a model crash |
| `_load_model()` verifies `.predict()` exists | Fails at startup, not on the first production request |
| `is_known_user()` / `is_known_movie()` | Cold start becomes assertable instead of inferred |
| `RootResponse`, `ModelInfoResponse`, `ModelMetrics`, `TrainingSetInfo` | Typed contracts replace raw dicts |
| `/model/info` reads `model_metadata.json` | A live instance can be asked how good its model was |
| `MAX_ID_LENGTH = 64` | Lab 1 used 64, the starter 50 — now one shared constant |
| `X-Process-Time-Ms` middleware | Per-request latency without a scrape |
| `responses={503, 500}` in OpenAPI | Error shapes are documented, not folklore |
| `app/cache.py` + Redis | Prediction cache, fails open |
| `app/metrics.py` + `/metrics` | Prometheus series for the existing dashboards |
| `docker-compose.yml`, `prometheus.yml` | The full stack, not just the API |

Four things Lab 3 does better were **kept**: the multi-stage Dockerfile, explicit
rating clipping in the wrapper, `reset_model()`/`reset_cache()` for test
isolation, and `lifespan` instead of loading the model at import time.

Porting these added 89 tests and 0 uncovered lines in `app/main.py`.

## Findings

Four things this lab surfaced that are worth recording:

**1. The starter's `test_client` fixture was broken.** It built `TestClient(app)`
without a context manager, so Starlette never ran the app lifespan, the model
global stayed `None`, and every `/predict` call returned 503 — the provided
`test_predict_valid_request_returns_200` could not pass. Fixed with
`with TestClient(app) as client:`.

**2. The suggested behavioural threshold fails on a correct model.** *"Every one
of 5 hand-picked pairs within 1.5 stars"* — one of those pairs (user 166 / movie
346, actual 1.0) is a user rating far below the crowd, and any collaborative
filter predicts ≈ 3.7 there. Gates were re-derived from the measured error
distribution (MAE 0.616, 94.5 % within 1.5, worst 2.81 on a 2,000-row sample) and
are now asserted over a fixed 500-row sample instead of 5 points.

**3. `scikit-surprise` has no wheels on PyPI.** Source-only, so it needs a C
compiler; unbuildable on Windows without MSVC. Hence the dual-backend design
above.

**4. The starter `.gitignore` silently excluded the data tests.** The line
`data/` is unanchored, so it matched `tests/data/` as well as the intended
data directory — the entire 48-test data package would never have been
committed, and CI would have run a suite with a whole pyramid level missing.
Changed to `/data/`.

---

## Deliverables

| Deliverable | Where | Status |
| --- | --- | --- |
| Test suite (unit / integration / data / model) | `tests/` | 278 passing |
| CI pipeline | `/.github/workflows/lab3-ci.yml` | lint → type → test(3.10, 3.11) → build |
| CD pipeline | `.github/workflows/cd.yml` | tag → Docker Hub + GHCR → staging → prod |
| Code quality setup | `.pre-commit-config.yaml`, `.flake8`, `pyproject.toml` | black · isort · flake8 · mypy clean |
| Coverage report | `htmlcov/` (CI artifact `coverage-report-py3.10`) | 100 % (≥ 80 % required) |
| Testing strategy | `docs/TESTING_STRATEGY.md` | |
| Passing-CI screenshots | [`docs/screenshots/`](docs/screenshots/) | with a note per image |
| Model quality gate | `scripts/validate_model.py` | RMSE < 1.0, MAE < 0.8 |

---

## Grading Rubric

| Criteria | Weight | Covered by |
| --- | --- | --- |
| Test coverage (unit, integration, data, model) | 30 % | 278 tests, all four levels, 100 % of `app/` |
| CI/CD pipeline | 30 % | 5-job CI with matrix + Docker smoke test; tag-driven CD |
| Code quality | 20 % | pre-commit pinned to CI versions; full type annotations |
| Documentation | 20 % | `docs/TESTING_STRATEGY.md`, this README, coverage report |
