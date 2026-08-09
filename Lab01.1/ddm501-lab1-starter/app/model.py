"""
ML Model wrapper for movie rating prediction.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained model from disk.

        Raises:
            FileNotFoundError: The .pkl file does not exist (run
                `python scripts/train_model.py` first).
            ValueError: The file exists but is not a usable model.
        """
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            logger.error(
                f"Model file not found: {self.model_path}. "
                "Run 'python scripts/train_model.py' to create it."
            )
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Model file is corrupted: {self.model_path} ({e})")
            raise ValueError(f"Could not unpickle model at {self.model_path}") from e

        if not hasattr(model, "predict"):
            raise ValueError(
                f"Object loaded from {self.model_path} has no predict() method "
                f"(got {type(model).__name__})"
            )

        self.model = model
        logger.info(f"Model loaded successfully from {self.model_path}")

    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Unknown users or movies do not raise: Surprise falls back to the global
        mean rating, so the API stays available for cold-start requests.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)

        Raises:
            ModelNotLoadedError: The model was never loaded.
            ValueError: user_id or movie_id is empty.
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded")

        if not str(user_id).strip() or not str(movie_id).strip():
            raise ValueError("user_id and movie_id must be non-empty strings")

        # Surprise clips the estimate to the trainset rating scale (1-5).
        prediction = self.model.predict(str(user_id), str(movie_id))
        return round(float(prediction.est), 2)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings
        """
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_known_user(self, user_id: str) -> bool:
        """Whether the user appears in the training set (False = cold start)."""
        try:
            self.model.trainset.to_inner_uid(str(user_id))
            return True
        except (ValueError, AttributeError):
            return False

    def is_known_movie(self, movie_id: str) -> bool:
        """Whether the movie appears in the training set (False = cold start)."""
        try:
            self.model.trainset.to_inner_iid(str(movie_id))
            return True
        except (ValueError, AttributeError):
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# =============================================================================
# Singleton instance (optional pattern)
# =============================================================================
# You can use this pattern to ensure only one model instance exists

_model_instance: Optional[MovieRatingModel] = None

def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
