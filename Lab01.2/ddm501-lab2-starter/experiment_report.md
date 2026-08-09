# Experiment Report — Movie Rating Prediction

_Generated 2026-08-09 04:05:21_

All runs share one train/test split (80/20, `random_state=42`) of MovieLens 100K, so the RMSE values below are directly comparable.

## Summary

- Total experiments: **9**
- Successful: **9**
- Failed: **0**
- Best RMSE: **0.9334** (`svd_n_factors=50_n_epochs=20_lr_all=0.005_reg_all=0.02`)

## All results

Sorted by RMSE, lower is better.

| # | Model | Parameters | RMSE | MAE | MSE | Train (s) | Run ID |
|--:|-------|------------|-----:|----:|----:|----------:|--------|
| 1 | SVD | n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02 | 0.9334 | 0.7356 | 0.8713 | 0.21 | `de43b723` |
| 2 | SVD | n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02 | 0.9392 | 0.7398 | 0.8822 | 0.48 | `6a6d66ee` |
| 3 | SVD | n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.02 | 0.9601 | 0.7529 | 0.9217 | 0.75 | `aad22eaf` |
| 4 | SVD | n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.02 | 0.9628 | 0.7548 | 0.9270 | 1.03 | `7aae1942` |
| 5 | KNN | k=40, name=pearson, user_based=True | 1.0150 | 0.8037 | 1.0303 | 0.23 | `d4c71ceb` |
| 6 | KNN | k=40, name=cosine, user_based=True | 1.0194 | 0.8038 | 1.0391 | 0.18 | `a7f98b52` |
| 7 | KNN | k=20, name=cosine, user_based=True | 1.0284 | 0.8099 | 1.0576 | 0.16 | `1161d346` |
| 8 | NMF | n_factors=50, n_epochs=50 | 1.0285 | 0.7858 | 1.0578 | 1.12 | `5751f385` |
| 9 | NMF | n_factors=100, n_epochs=50 | 1.1017 | 0.8380 | 1.2137 | 1.90 | `a3cece45` |

## Best model

- Run name: `svd_n_factors=50_n_epochs=20_lr_all=0.005_reg_all=0.02`
- Run ID: `de43b723c80a49a2b87faa70f15c488c`
- Configuration: `{"lr_all": 0.005, "model_type": "svd", "n_epochs": 20, "n_factors": 50, "reg_all": 0.02}`

| Metric | Value |
|--------|------:|
| rmse | 0.9334 |
| mae | 0.7356 |
| mse | 0.8713 |
| mape | 29.7470 |
| coverage | 1.0000 |
| training_time_seconds | 0.2104 |

## Analysis

### Algorithm comparison

| Algorithm | Runs | Best RMSE | Worst RMSE | Mean RMSE |
|-----------|-----:|----------:|-----------:|----------:|
| SVD | 4 | 0.9334 | 0.9628 | 0.9489 |
| KNN | 3 | 1.0150 | 1.0284 | 1.0209 |
| NMF | 2 | 1.0285 | 1.1017 | 1.0651 |

SVD beats NMF by 0.0951 RMSE (9.2%) at each family's best setting.

### Effect of n_factors (SVD)

| n_factors | Best RMSE |
|----------:|----------:|
| 50 | 0.9334 |
| 100 | 0.9392 |
| 150 | 0.9601 |

Going from 50 to 150 factors degrades RMSE by 0.0266. More capacity is not automatically better here — MovieLens 100K is small enough that large factor counts overfit unless regularisation rises with them.

### Training cost

- Fastest: `knn_k=20_name=cosine_user_based=True` at 0.16s (RMSE 1.0284)
- Slowest: `nmf_n_factors=100_n_epochs=50` at 1.90s (RMSE 1.1017)
- Best RMSE: `svd_n_factors=50_n_epochs=20_lr_all=0.005_reg_all=0.02` at 0.21s

### Coverage (cold-start fallbacks)

| Algorithm | Coverage | Cold-start pairs |
|-----------|---------:|-----------------:|
| KNN | 0.9971 | 58 |
| KNN | 0.9982 | 36 |
| NMF | 0.9982 | 36 |
| SVD | 1.0000 | 0 |

Coverage ranges from 0.9971 to 1.0000. The 58 pairs that `knn_k=40_name=pearson_user_based=True` could not score fell back to the global mean, so its RMSE is very slightly flattered — the fallback is a safe average rather than a real, riskier prediction. At 0.29% of the test set the effect is negligible here, but the same gap on a sparser catalogue would make RMSE misleading.

## Recommendation

Deploy **SVD** with `{"lr_all": 0.005, "n_epochs": 20, "n_factors": 50, "reg_all": 0.02}`. It has the lowest RMSE (0.9334) on the shared test split and trains in 0.21s, which fits comfortably inside a weekly retraining window.

The Airflow DAG registers this configuration automatically: it promotes the best run to the `Production` stage only when RMSE is below 1.0, so a degraded retrain cannot silently replace a good model.
