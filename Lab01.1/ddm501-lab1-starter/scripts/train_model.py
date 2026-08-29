"""
Script to train and save the movie rating prediction model.

This script:
1. Downloads the MovieLens 100K dataset
2. Trains an SVD model using collaborative filtering
3. Evaluates the model with cross-validation
4. Saves the trained model to disk

Usage:
    python scripts/train_model.py
"""

import json
import pickle
import os
from datetime import datetime, timezone
from pathlib import Path

from surprise import Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy


def main():
    """Main function to train and save the model."""
    
    print("=" * 60)
    print("Movie Rating Prediction Model Training")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "svd_model.pkl"
    
    # ==========================================================================
    # Step 1: Load data
    # ==========================================================================
    print("\n[1/4] Loading MovieLens 100K dataset...")
    # CI and containers do not have an interactive stdin. Surprise asks before
    # downloading by default, which makes a clean automated run fail with EOF.
    data = Dataset.load_builtin('ml-100k', prompt=False)
    print("      Dataset loaded successfully!")
    print(f"      - This dataset contains 100,000 ratings")
    print(f"      - From 943 users on 1,682 movies")
    
    # ==========================================================================
    # Step 2: Cross-validation
    # ==========================================================================
    print("\n[2/4] Performing cross-validation...")
    
    # Define the SVD model
    model = SVD(
        n_factors=100,      # Number of latent factors
        n_epochs=20,        # Number of training epochs
        lr_all=0.005,       # Learning rate
        reg_all=0.02        # Regularization term
    )
    
    # Run 5-fold cross-validation
    cv_results = cross_validate(
        model, 
        data, 
        measures=['RMSE', 'MAE'], 
        cv=5, 
        verbose=True
    )
    
    print(f"\n      Cross-validation results:")
    print(f"      - Mean RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"      - Mean MAE:  {cv_results['test_mae'].mean():.4f}")
    
    # ==========================================================================
    # Step 3: Train on full dataset
    # ==========================================================================
    print("\n[3/4] Training on full dataset...")
    
    trainset = data.build_full_trainset()
    model.fit(trainset)
    
    print("      Training completed!")
    
    # ==========================================================================
    # Step 4: Save model
    # ==========================================================================
    print(f"\n[4/4] Saving model to {model_path}...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("      Model saved successfully!")

    # Metrics are served by GET /model/info, so persist them next to the model.
    metadata_path = models_dir / "model_metadata.json"
    metadata = {
        "model_type": "SVD (Collaborative Filtering)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "rmse": round(float(cv_results['test_rmse'].mean()), 4),
            "mae": round(float(cv_results['test_mae'].mean()), 4),
            "cv_folds": 5,
        },
        "training_set": {
            "dataset": "MovieLens 100K",
            "n_users": trainset.n_users,
            "n_items": trainset.n_items,
            "n_ratings": trainset.n_ratings,
        },
        "hyperparameters": {
            "n_factors": 100,
            "n_epochs": 20,
            "lr_all": 0.005,
            "reg_all": 0.02,
        },
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"      Metadata saved to {metadata_path}")
    
    # ==========================================================================
    # Test prediction
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Testing the model...")
    print("=" * 60)
    
    # Test prediction for user 196 and movie 242
    test_user = "196"
    test_movie = "242"
    prediction = model.predict(test_user, test_movie)
    
    print(f"\nSample prediction:")
    print(f"  User ID:           {test_user}")
    print(f"  Movie ID:          {test_movie}")
    print(f"  Predicted Rating:  {prediction.est:.2f}")
    
    print("\n" + "=" * 60)
    print("Training complete! You can now run the API.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start the API: uvicorn app.main:app --reload")
    print("  2. Open docs:     http://localhost:8000/docs")
    print("  3. Test predict:  POST /predict with user_id and movie_id")


if __name__ == "__main__":
    main()
