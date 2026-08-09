"""
ML Model wrapper for movie rating prediction.
"""

import logging
import pickle
from typing import Any, List, Optional, Tuple

from app.config import MAX_RATING, MIN_RATING, MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """
    Raised when a prediction is requested before the model is available.

    Subclasses RuntimeError so existing callers that catch RuntimeError keep
    working, while the API layer can map this specific case to 503 rather than
    to a generic 500.
    """


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    - Reporting whether an ID was seen during training (cold-start detection)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path: str = model_path
        self.model: Optional[Any] = None
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
            with open(self.model_path, "rb") as f:
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

        # A pickle can contain anything. Fail here, at load time, rather than on
        # the first request in production.
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

        Unknown users or movies do not raise: the model falls back to the global
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
        algorithm = self.model
        if algorithm is None:
            raise ModelNotLoadedError("Model not loaded")

        if not str(user_id).strip() or not str(movie_id).strip():
            raise ValueError("user_id and movie_id must be non-empty strings")

        prediction = algorithm.predict(str(user_id), str(movie_id))
        rating = float(round(prediction.est, 2))

        # Clip to the valid range. The backends already do this, but the API
        # response schema enforces 1-5 and a schema violation would be a 500.
        return max(MIN_RATING, min(MAX_RATING, rating))

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings, in the order of the input pairs
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Model not loaded")

        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_known_user(self, user_id: str) -> bool:
        """Whether the user appears in the training set (False = cold start)."""
        trainset = getattr(self.model, "trainset", None)
        if trainset is None:
            return False
        try:
            trainset.to_inner_uid(str(user_id))
            return True
        except (ValueError, KeyError, AttributeError):
            return False

    def is_known_movie(self, movie_id: str) -> bool:
        """Whether the movie appears in the training set (False = cold start)."""
        trainset = getattr(self.model, "trainset", None)
        if trainset is None:
            return False
        try:
            trainset.to_inner_iid(str(movie_id))
            return True
        except (ValueError, KeyError, AttributeError):
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Singleton instance
_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance


def reset_model() -> None:
    """Reset the model instance (useful for testing)."""
    global _model_instance
    _model_instance = None
