# Testing Strategy — Movie Rating Prediction API

DDM501 · Lab 3 · Testing & CI/CD

---

## 1. Why testing an ML system is different

A traditional service is correct when the code is correct. An ML service can have
perfectly correct code and still be wrong, because two extra things can break
independently:

| What can break | Caught by | Example failure |
| --- | --- | --- |
| Code | unit / integration tests | `predict()` returns a numpy scalar the schema rejects |
| Data | data tests | the ratings file silently truncates to 40k rows |
| Model | behavioural tests | retraining produces a constant predictor |

All three are in scope here. A suite that only covers the first column would go
green while the service returns 3.53 for every user.

---

## 2. The test pyramid as implemented

```
                    ┌───────────────────────────────┐
                    │  Behavioural   31 tests       │  slowest, needs an artifact
                    │  tests/model/                 │
                ┌───┴───────────────────────────────┴───┐
                │  Data quality      48 tests           │
                │  tests/data/                          │
            ┌───┴───────────────────────────────────────┴───┐
            │  Integration         109 tests                │
            │  tests/integration/  (api 54, cache 30,       │
            │                       metrics 25)             │
        ┌───┴───────────────────────────────────────────────┴───┐
        │  Unit                    90 tests                     │  fastest, no I/O
        │  tests/unit/                                          │
        └───────────────────────────────────────────────────────┘

                    278 tests · 2.4 s in CI · 100 % coverage of app/
```

The whole suite runs in about **5 seconds**. That is deliberate: a suite slow
enough to skip is a suite that gets skipped.

Five tests need a live Redis and skip locally; CI supplies one as a service
container, which also covers the last two lines of `app/cache.py`.

### 2.1 Unit tests — `tests/unit/` (90)

Scope: one class or one schema at a time, no network, no real artifact where it
can be avoided.

- **`test_model.py`** — the `MovieRatingModel` wrapper: load success, load
  failure (missing file, corrupted pickle), return type, rounding, range
  clipping, batch length/order, `is_loaded()`, `RuntimeError` when the algorithm
  is absent, and the `get_model()`/`reset_model()` singleton.
  Range clipping is tested with a **pickled stub algorithm** returning `9.9` and
  `-4.2`, because a real model never produces those — and untested clipping is
  exactly how an out-of-range rating reaches a client.
- **`test_schemas.py`** — the request/response contract: required fields,
  `min_length`/`max_length` boundaries (49/50/51), whitespace-only rejection,
  whitespace stripping, `None` rejection, int-is-not-str, batch size bounds
  (0/1/100/101), and the `1.0`/`5.0` response boundaries.

Boundaries are tested on **both sides plus the edge itself**; off-by-one in a
validator is the single most common way bad data reaches a model.

Two capabilities restored from Lab 1 earn their own tests here:

- **Cold-start detection** (`is_known_user` / `is_known_movie`). Previously a
  cold start could only be *inferred* from a differing prediction. Now
  `test_zero_padded_id_is_not_the_same_user` asserts directly that `"0196"` is
  not `"196"` — a fact a client integrating against this API needs to know.
- **Load-time artifact validation**. A pickle can contain anything;
  `test_pickle_without_predict_method_is_rejected` proves the wrapper refuses a
  plain dict at startup rather than crashing on the first production request.

### 2.2 Integration tests — `tests/integration/` (109)

Scope: real HTTP requests through Starlette's `TestClient` against the full app —
routing, middleware, schema validation, model wrapper, response serialisation.

Covered: `/`, `/health`, `/model/info`, `/predict`, `/predict/batch`, the OpenAPI
document, CORS headers, and every error path —

| Condition | Expected | Why it matters |
| --- | --- | --- |
| Missing/blank/whitespace field | `422` | client error, reported per field |
| Malformed JSON body | `422` | must not surface as a 500 |
| Model raises `ValueError` on IDs | `422` | still a caller mistake, even found late |
| `GET /predict`, `POST /health` | `405` | method contract |
| Unknown route | `404` | |
| Model not loaded | **`503`** | tells a load balancer to route elsewhere; a 500 does not |
| Model vanishes mid-request | **`503`** | an artifact swapped under a live worker is retryable |
| Model raises at inference | `500` | failure is contained, not leaked as a crash |

The last two are produced by `monkeypatch`-ing `app.main.model` — to `None`, and
to a stub that raises. Without them the whole error-handling half of `main.py`
would be untested, which is how a "healthy" service returns 200 with no model
behind it.

**One bug was found and fixed here.** The starter's `test_client` fixture built
`TestClient(app)` without a context manager, so Starlette never ran the
application lifespan, the model global stayed `None`, and every `/predict` call
returned 503. The fixture now uses `with TestClient(app) as client:`.

#### Cache tests — `test_cache.py` (30)

The cache's defining property is that it **fails open**: if Redis is missing,
slow, or broken, the API keeps serving and only loses the speedup. A cache that
turns a dependency outage into an API outage is worse than no cache at all, so
that property is what these tests are for.

| Scenario | Required behaviour |
| --- | --- |
| Cache disabled (the default) | `redis` is never even imported |
| Unreachable Redis at startup | construction degrades, does not raise |
| Read failure | looks like a miss |
| Write failure | swallowed; the prediction still returns |
| Corrupted (non-numeric) entry | treated as a miss |
| After any failure | backoff — no further Redis calls for 30 s |
| Redis down | `/health` still `healthy`; only `cache_connected` is false |

The failure branches are driven by a stub client, because a live Redis cannot be
asked to fail on command. Round-trip behaviour is tested against a real Redis
when one is reachable.

The most important test in the file is `test_cache_hit_skips_the_model`: it
asserts the model is **not** called on a hit. Without it the cache could be
silently dead — every request would still return the right answer, just slowly,
and nothing would notice.

Key construction is tested too: `test_key_includes_model_version` pins that
retraining starts with a cold cache instead of inheriting the previous model's
answers.

#### Metrics tests — `test_metrics.py` (25)

Monitoring is only worth having if it is correct; an endpoint that returns 200
with the wrong labels produces dashboards that lie.

- **Format** — the Prometheus content type (serving JSON here would make the
  endpoint unscrapeable while still returning 200), and `# HELP`/`# TYPE` lines.
- **Contents** — every series a dashboard depends on, checked by name.
- **Counters actually move** — including on `422`s. An error-rate panel that
  silently drops errors is the worst kind of monitoring bug.
- **Label cardinality** — labels carry `/predict`, never `/predict?user=196`.
  Labelling raw URLs would create a new time series per URL, which is unbounded
  cardinality and the classic way to melt a Prometheus server. `/metrics` also
  excludes itself, so scrapes do not inflate what they report.
- **Gauges** — `model_loaded` drops to 0 when the model disappears. That is the
  series an alert fires on.

### 2.3 Data quality tests — `tests/data/` (48)

Two halves, both necessary.

**Half 1 — the validators** (`training/validators.py`): schema, dtypes,
completeness (nulls *and* blank strings), value range, uniqueness, row count,
and mean-in-band. Each validator is asserted against clean data **and** against
deliberately corrupted data from the `corrupted_ratings` fixture.

> A validator that is only ever shown clean data is indistinguishable from a
> validator that does nothing. `test_corrupted_dataset_reports_multiple_failures`
> is the test that proves the suite has teeth.

**Half 2 — the real MovieLens 100K data**, validated against the contract the
training pipeline assumes:

| Check | Expectation |
| --- | --- |
| Row count | exactly 100,000 |
| Distinct users / movies | 943 / 1,682 |
| Missing values | none |
| Rating range | `[1, 5]`, whole stars only |
| Duplicate `(user, movie)` | none |
| Mean rating | within `[3.4, 3.7]` (historical ≈ 3.53) |
| Ratings per user | ≥ 20 |
| ID dtype | string, matching the API schema |

The mean-rating band is the cheapest useful **drift detector**: if the dataset is
swapped or resampled, the mean moves and the build goes red before training
starts.

`test_known_pairs_exist_in_dataset` closes a subtle hole — it asserts the pairs
the behavioural tests rely on really are in the training data. Without it, a
"prediction close to actual" test could be silently measuring cold-start
behaviour instead.

These tests **skip** (not fail) when the dataset is not cached locally, so the
suite stays runnable offline.

> **A second bug was found here.** The starter `.gitignore` contained an
> unanchored `data/`, which matches `tests/data/` as well as the intended data
> directory. The entire data-test package would never have been committed and CI
> would have run a suite missing a whole pyramid level, while still reporting
> green. Changed to `/data/`.

### 2.4 Behavioural tests — `tests/model/` (31)

Scope: what the model *does*, not which lines it executes.

**Invariance** — the output must not change under perturbations that carry no
meaning:

- same input → same output (5 repeated calls, and across all known pairs)
- batch order → results permute, never change
- individual vs batch → identical
- batch size → identical (a pair alone vs padded into a batch of 21)
- pickle round-trip → a reloaded model predicts identically
- 100 intervening calls do not mutate state

**Directional** — the output must move the way domain knowledge says it should:

- movie 50 (*Star Wars*, mean ≈ 4.36) scores **above** movie 122 (*Cape Fear*,
  mean ≈ 2.6) for the same user → item biases learned in the right direction
- user 4 (generous, mean ≈ 4.3) scores **above** user 181 (harsh, mean ≈ 1.5) on
  the same movie → user biases learned in the right direction
- an unknown user falls back **near the global mean** (`2.5–4.5`), not to an
  extreme — a cold-start prediction of 1.0 is a confident wrong answer

**Minimum functionality** — cases that must work for the model to ship:

- a known pair predicts in range
- predictions are **not all identical** (a constant predictor passes every range
  check and is useless)
- the spread across 10 movies exceeds 0.3 stars (guards against collapse onto
  the global mean)

**Robustness** — unknown users/movies, both unknown, 200-character IDs,
zero-padded IDs, and injection-shaped IDs (`'; DROP TABLE ratings;--`,
`../../etc/passwd`, `<script>`, embedded NUL). Every one must return a valid
rating or a typed error — never garbage, never a crash.

**Performance gates** — MAE, the fraction within 1.5 stars, the worst single
error, the RMSE recorded at training time, and throughput (1,000 predictions
under 5 s, guarding against an accidental model reload per prediction).

---

## 3. Calibrating the quality gates

The starter's suggested gate — *every one of 5 hand-picked pairs within 1.5
stars* — **fails against a correctly trained model**, and would fail with
scikit-surprise too. One of those pairs (user 166 / movie 346, actual 1.0) is a
user rating far below the crowd; any collaborative-filtering model predicts ≈ 3.7
there. A 5-point gate measures sampling luck, not model quality.

Gates were therefore re-derived from the measured error distribution on a
2,000-row sample of the training set:

| Statistic | Measured | Gate | Headroom |
| --- | --- | --- | --- |
| MAE | 0.616 | < 0.85 | 38 % |
| Within 1.5 stars | 94.5 % | ≥ 85 % | 11 pts |
| Worst single error | 2.81 | < 3.5 | 25 % |
| Training RMSE | 0.944 | < 1.00 | 6 % |
| Training MAE | 0.745 | < 0.80 | 7 % |

Each gate is asserted over a **fixed 500-row random sample** (`random_state=42`)
so it is reproducible across machines, and is loose enough that ordinary
retraining variance does not turn the build red — while a genuinely broken
artifact (untrained, shuffled index, constant output) trips it immediately.

The same numbers back `scripts/validate_model.py`, which fails the CI build on a
metrics regression *before* any image is built.

---

## 4. Fixture design

Defined in `tests/conftest.py`:

| Fixture | Scope | Purpose |
| --- | --- | --- |
| `test_client` | session | `TestClient` inside a context manager (runs lifespan) |
| `api_client` | function | `test_client`, but **skips** if no model is loaded |
| `trained_model` | session | model loaded once, reused by 95 tests |
| `movielens_ratings` | session | raw dataset, skips when not cached |
| `evaluation_sample` | session | fixed 500-row sample for quality gates |
| `sample_ratings` / `ratings_dataframe` | function | clean data for validators |
| `corrupted_ratings` | function | broken data, to prove validators fire |
| `known_user_movie_pairs` | function | real pairs with true ratings |
| `unknown_users` / `unknown_movies` | function | cold-start inputs |

Session scope for anything expensive (model load, 100k-row CSV parse); function
scope for anything mutable, so no test can contaminate another.

Tests **skip rather than fail** when a prerequisite artifact is missing — a
missing model is a setup problem, not a defect, and conflating the two teaches
people to ignore red builds.

---

## 5. Coverage

As reported by CI (`Run Tests (py3.10)`, run #9 — see
[`docs/screenshots/03-ci-coverage-output.png`](screenshots/03-ci-coverage-output.png)):

```
---------- coverage: platform linux, python 3.10.20-final-0 ----------
Name              Stmts   Miss  Cover   Missing
-----------------------------------------------
app/__init__.py       1      0   100%
app/cache.py         71      0   100%
app/config.py        20      0   100%
app/main.py         114      0   100%
app/metrics.py       24      0   100%
app/model.py         66      0   100%
app/schemas.py       76      0   100%
-----------------------------------------------
TOTAL               372      0   100%

Required test coverage of 80% reached. Total coverage: 100.00%
========================= 278 passed in 2.41s =========================
```

**100 %** of `app/`, against a required minimum of 80 %. The gate is enforced
twice — `fail_under = 80` in `pyproject.toml` and `--cov-fail-under=80` in the
workflow, so it is visible in the CI log.

Run the same suite on a laptop with no Redis and it reports **273 passed, 5
skipped, 99 %**: `app/cache.py` keeps two uncovered lines, the successful
Redis-connect path, which no stub can reach. CI supplies a Redis service
container, and those two lines close. Both numbers are honest — the difference
*is* the point, and it is why the cache tests skip rather than fail when Redis
is absent.

Coverage is a *floor*, not a goal. `app/` reaching 100 % says every line runs;
the behavioural and data tests are what say the system is *right*. `training/`
is intentionally outside the coverage target — it builds artifacts, it does not
serve them.

---

## 6. CI/CD pipeline

```
  lint ─────┐
            ├──> test (3.10, 3.11) ──> build ──> ci-status
  type-check┘     ├ train model
                  ├ validate model quality  ← fails on metric regression
                  └ pytest + coverage gate
```

Cheap checks gate expensive ones: lint and mypy finish in seconds and stop a bad
commit before anyone pays for a training run. The matrix covers Python 3.10 and
3.11. `concurrency` cancels superseded runs.

The test job runs a **Redis service container**, so the cache round-trip tests
execute instead of skipping. The API never requires Redis — the cache fails open
— but an untested cache is a cache nobody should trust.

The build job downloads the model trained by the test job (rather than
retraining) and smoke-tests the container on `"model_loaded":true` — **a build
that succeeds but serves 503 is still a broken release**.

`ci-status` aggregates everything into one required check for branch protection.

CD (`cd.yml`, on `v*` tags): re-verify → build multi-arch and push to Docker Hub
+ GHCR → GitHub release → staging → production behind a manual approval
environment, with an explicit rollback step.

---

## 7. Code quality

| Tool | Configured in | Enforced by |
| --- | --- | --- |
| black (line length 100) | `pyproject.toml` | pre-commit + CI |
| isort (black profile) | `pyproject.toml` | pre-commit + CI |
| flake8 | `.flake8` | pre-commit + CI |
| mypy (`warn_return_any`) | `pyproject.toml` | pre-commit + CI |

Pre-commit pins the **same versions** as CI, so a commit that passes locally
passes in CI — a formatting disagreement between the two is impossible by
construction. Hooks are split by cost: unit + data tests on `pre-commit`
(< 1 s), the full suite with the coverage gate on `pre-push`.

`app/`, `training/` and `scripts/` are fully type-annotated and mypy-clean.

---

## 8. What is deliberately not tested here

Honest scope boundaries:

- **Load / stress testing** — needs a real deployment; `test_batch_throughput`
  is a smoke check, not a load test.
- **Production drift monitoring** — the mean-rating band catches drift in the
  *training* data. The `/metrics` endpoint publishes the deployed model's RMSE
  and MAE next to live traffic, but a live *prediction-distribution* monitor
  (and the alert rules on top of it) is not built here.
- **Fairness / bias audits** — MovieLens 100K carries demographics that would
  make this meaningful; out of scope for this lab.
- **A/B or shadow evaluation** — the model gate compares against fixed
  thresholds, not against the currently deployed model. A champion/challenger
  comparison is the natural next step.
