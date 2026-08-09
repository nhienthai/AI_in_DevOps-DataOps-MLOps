"""
Model validation gate for CI.

Fails the build when a freshly trained model does not meet the minimum quality
bar, so a regression is caught before the image is ever built. This is the
"model validation" stage of an ML CI pipeline: unit tests prove the code runs,
this proves the artifact is worth shipping.

Usage:
    python scripts/validate_model.py
    python scripts/validate_model.py --max-rmse 0.95 --max-mae 0.80
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
MODEL_PATH = PROJECT_ROOT / "models" / "svd_model.pkl"

DEFAULT_MAX_RMSE = 1.00
DEFAULT_MAX_MAE = 0.80


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate the trained model against gates.")
    parser.add_argument("--max-rmse", type=float, default=DEFAULT_MAX_RMSE)
    parser.add_argument("--max-mae", type=float, default=DEFAULT_MAX_MAE)
    return parser.parse_args()


def main() -> int:
    """Return 0 when the model clears every gate, 1 otherwise."""
    args = parse_args()

    failures = []

    if not MODEL_PATH.exists():
        print(f"FAIL  model artifact missing: {MODEL_PATH}")
        return 1

    if not METRICS_PATH.exists():
        print(f"FAIL  metrics file missing: {METRICS_PATH}")
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    print("=" * 60)
    print("Model validation")
    print("=" * 60)

    for name, gate in (("rmse", args.max_rmse), ("mae", args.max_mae)):
        value = float(metrics[name])
        status = "PASS" if value < gate else "FAIL"
        print(f"{status}  {name.upper():5s} = {value:.4f}  (gate: < {gate})")
        if status == "FAIL":
            failures.append(f"{name}={value:.4f} >= {gate}")

    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"INFO  artifact size = {size_mb:.2f} MB")

    print("=" * 60)
    if failures:
        print("Model validation FAILED: " + "; ".join(failures))
        return 1
    print("Model validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
