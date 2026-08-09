# Submission screenshots — Lab 2

The lab requires *"screenshots of MLflow UI showing experiments"*. This folder
holds them, with a note per image saying what it proves.

Naming: `NN-surface-what.png`, numbered in the order a reader should look at
them. Keep the numbers — they are how the notes below line up with the files.

> `.gitignore` blocks `*.png` project-wide (generated plots land in
> `artifacts/`). It carries an explicit exception for this folder, so these
> images do get committed. Verify with `git check-ignore -v docs/screenshots/<file>.png`
> — no output means the file is tracked.

---

## Captured

The MLflow **Chart** tab of the `hyperparameter-tuning` experiment, 9 runs
compared side by side. All five were taken at the same moment, so the numbers
are consistent across them.

Numbered in the order the charts appear when scrolling the Chart tab, so saving
them is mechanical: first capture → `01`, second → `02`, and so on.

| # | File | Shows | Reads as |
|--:|------|-------|----------|
| 1 | `01-mlflow-chart-coverage-trainingtime.png` | `coverage` and `training_time_seconds` | Training cost spans 0.07 s (KNN) to 1.18 s (NMF). The best model, SVD `n_factors=50`, trains in 0.14 s — cheapest *and* most accurate. |
| 2 | `02-mlflow-chart-rmse-mae.png` | `rmse` and `mae` bar charts | SVD sweeps 0.93–0.97; KNN and NMF all sit at or above 1.02. The ranking is visible without reading a single number. |
| 3 | `03-mlflow-chart-mse-rmse-manual.png` | `mse` and `rmse_manual` | `rmse_manual` matches `rmse` bar for bar — an independent recomputation of the metric agreeing with Surprise's own. |
| 4 | `04-mlflow-chart-mape-npredictions.png` | `mape` and `n_predictions` | MAPE 29.8–37.3%. `n_predictions` is 20 000 for every run — proof all nine were scored on the *same* test split. |
| 5 | `05-mlflow-chart-npredictions-nimpossible.png` | `n_predictions` and `n_impossible` | 4 SVD runs have 0 cold-start fallbacks; KNN/NMF have 36, one KNN has 58. |

### One caveat on reading these charts

The `coverage` chart in **01** displays `1.00` for all nine runs, which looks
like perfect coverage. It is not — MLflow rounds the bar labels to two decimals.
The true values are:

| Algorithm | Coverage | Cold-start pairs |
|-----------|---------:|-----------------:|
| KNN `k=40, pearson` | 0.9971 | 58 |
| KNN / NMF (others) | 0.9982 | 36 |
| SVD (all four) | 1.0000 | 0 |

Screenshot **05** is the honest one: `n_impossible` shows the 58 and the 36s
plainly, because integers do not round away. When these two charts disagree,
trust `n_impossible`.

This is worth knowing beyond this lab: a rounded chart label is not the value.
The same rounding once put a wrong sentence into `experiment_report.md` —
"coverage is 1.000 across all runs" — which is now corrected.

---

## Still to capture

The five above are all the same view. The rubric also credits parameter
logging, artifacts and the model registry, and none of those appear in a chart.
Add these before submitting:

| # | File to add | Where | Why it is needed |
|--:|-------------|-------|------------------|
| 6 | `06-mlflow-table-comparison.png` | `hyperparameter-tuning` → **Table** tab, show the params columns | **The single most important shot.** It is the only view showing hyperparameters and metrics on the same row — the point of experiment tracking. Charts show metrics only. |
| 7 | `07-mlflow-run-detail.png` | Click the best run → its detail page | Parameters (5%) and metrics (5%) on the rubric, in one frame. |
| 8 | `08-mlflow-run-artifacts.png` | Same run → **Artifacts** tab, expand `model/` and `plots/` | Artifacts (4%). Shows `model_svd_*.pkl` plus both PNGs. |
| 9 | `09-mlflow-model-registry.png` | Top nav → **Models** → `movie-rating-model` | Model registry (3%). Shows versions and which one is in `Production`. |
| 10 | `10-mlflow-experiment-list.png` | Experiments sidebar | Both experiments exist: `movie-rating-prediction` and `hyperparameter-tuning`. |
| 11 | `11-airflow-dag-graph.png` | Airflow → `movie_rating_training` → **Graph** | Airflow automation is 20% of the grade, and no MLflow screenshot covers it. Shows the branch splitting to `register_model` / `skip_registration`. |
| 12 | `12-airflow-dag-runs.png` | Same DAG → **Grid** | A green run, and the `@weekly` schedule. Capture it with both the manual and the scheduled run visible. |

### How to get the stack back up

```bash
cd ddm501-lab2-starter
docker compose up -d
```

| Surface | URL | Login |
|---------|-----|-------|
| MLflow | <http://localhost:5001> | none |
| Airflow | <http://localhost:8080> | `admin` / `admin` |

If MLflow shows no runs, the experiments have not been run in this environment:

```bash
python -m experiments.run_experiments
```

To regenerate only the report — after a reporting fix, without retraining and
without duplicating the nine runs already in the UI:

```bash
python -m experiments.run_experiments --report-only
```

### Tips for the remaining shots

- On the **Table** tab, use the columns selector to show `model_type`,
  `n_factors`, `n_epochs`, `rmse`, `mae`. The default column set hides params.
- Sort by `rmse` ascending so the best run is the top row.
- Leave the search box empty. The grey text in it
  (`metrics.rmse < 1 and params.model = "tree"`) is MLflow's placeholder
  example, not an applied filter — "9 matching runs" at the bottom left
  confirms nothing is filtered out.
- Full browser window, no zoom below 100%, so the numbers stay readable when
  the grader views the file.
