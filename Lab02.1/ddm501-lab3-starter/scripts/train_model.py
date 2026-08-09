"""
Script to train and save the movie rating prediction model.

Two interchangeable backends produce the same artifact contract
(``model.predict(uid, iid).est``):

* ``surprise`` - ``scikit-surprise``'s SVD, the reference implementation.
* ``local``    - the NumPy-only :class:`training.local_svd.LocalSVD`, used when
  scikit-surprise cannot be built (e.g. CPython 3.12 on Windows without MSVC).

``--backend auto`` (the default) picks ``surprise`` when it imports, otherwise
falls back to ``local``, so the same command works on a laptop and in CI.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --backend local --cv 3
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

# Make the project root importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import dataset  # noqa: E402
from training.local_svd import LocalSVD, mae, rmse  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "svd_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

HYPERPARAMETERS: Dict[str, Any] = {
    "n_factors": 50,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
}


def surprise_available() -> bool:
    """Return True when scikit-surprise can be imported."""
    try:
        import surprise  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(requested: str) -> str:
    """Turn ``auto`` into a concrete backend name and validate explicit choices."""
    if requested == "auto":
        return "surprise" if surprise_available() else "local"
    if requested == "surprise" and not surprise_available():
        raise SystemExit(
            "Backend 'surprise' requested but scikit-surprise is not installed.\n"
            "Install it (needs a C compiler) or run with --backend local."
        )
    return requested


# =============================================================================
# Backends
# =============================================================================


def train_with_surprise(cv_folds: int) -> Tuple[Any, Dict[str, float]]:
    """Train scikit-surprise's SVD and return ``(model, metrics)``."""
    from surprise import SVD, Dataset
    from surprise.model_selection import cross_validate

    print("\n[1/4] Loading MovieLens 100K dataset (scikit-surprise)...")
    # prompt=False matters in CI: the interactive download prompt would hang.
    data = Dataset.load_builtin("ml-100k", prompt=False)
    print("      Dataset loaded successfully!")

    model = SVD(**HYPERPARAMETERS, random_state=42)

    print(f"\n[2/4] Performing {cv_folds}-fold cross-validation...")
    cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=cv_folds, verbose=False)
    metrics = {
        "rmse": float(cv_results["test_rmse"].mean()),
        "mae": float(cv_results["test_mae"].mean()),
        "cv_folds": cv_folds,
    }

    print("\n[3/4] Training on full dataset...")
    model.fit(data.build_full_trainset())
    return model, metrics


def train_with_local(cv_folds: int) -> Tuple[LocalSVD, Dict[str, float]]:
    """Train the NumPy fallback SVD and return ``(model, metrics)``."""
    print("\n[1/4] Loading MovieLens 100K dataset (local backend)...")
    triplets = dataset.load_triplets()
    print(f"      Loaded {len(triplets):,} ratings.")

    print(f"\n[2/4] Performing {cv_folds}-fold cross-validation...")
    import numpy as np

    rng = np.random.default_rng(42)
    folds = rng.integers(0, cv_folds, size=len(triplets))
    rmses, maes = [], []
    for fold in range(cv_folds):
        train = [t for t, f in zip(triplets, folds) if f != fold]
        test = [t for t, f in zip(triplets, folds) if f == fold]
        fold_model = LocalSVD(**HYPERPARAMETERS, random_state=42).fit(train)
        predictions = fold_model.test(test)
        rmses.append(rmse(predictions))
        maes.append(mae(predictions))
        print(f"      fold {fold + 1}/{cv_folds}: RMSE={rmses[-1]:.4f} MAE={maes[-1]:.4f}")

    metrics = {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "cv_folds": cv_folds,
    }

    print("\n[3/4] Training on full dataset...")
    model = LocalSVD(**HYPERPARAMETERS, random_state=42).fit(triplets)
    return model, metrics


# =============================================================================
# Entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the movie rating model.")
    parser.add_argument(
        "--backend",
        choices=["auto", "surprise", "local"],
        default="auto",
        help="Training backend (default: auto).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Train, evaluate and persist the model plus its metadata."""
    args = parse_args()
    backend = resolve_backend(args.backend)

    print("=" * 60)
    print("Movie Rating Prediction Model Training")
    print(f"Backend: {backend}")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    if backend == "surprise":
        model, metrics = train_with_surprise(args.cv)
    else:
        model, metrics = train_with_local(args.cv)

    print(f"      Mean RMSE: {metrics['rmse']:.4f}")
    print(f"      Mean MAE:  {metrics['mae']:.4f}")

    print(f"\n[4/4] Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_type": "SVD (Collaborative Filtering)",
                "backend": backend,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "training_set": {
                    "dataset": "MovieLens 100K",
                    "n_users": dataset.EXPECTED_N_USERS,
                    "n_items": dataset.EXPECTED_N_MOVIES,
                    "n_ratings": dataset.EXPECTED_N_RATINGS,
                },
                "hyperparameters": HYPERPARAMETERS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("      Model, metrics and metadata saved successfully!")

    print("\n" + "=" * 60)
    prediction = model.predict("196", "242")
    print(f"Sample prediction for user 196, movie 242: {prediction.est:.2f}")
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
