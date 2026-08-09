"""
Unit tests for MovieRatingModel class.

Level 1 of the ML testing pyramid: the model *wrapper* in isolation - loading,
return types, range clipping, batching and failure modes. Prediction quality is
not asserted here; that belongs to tests/model/test_model_behavior.py.

Run tests:
    pytest tests/unit/test_model.py -v
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.config import MAX_RATING, MIN_RATING
from app.model import ModelNotLoadedError, MovieRatingModel, get_model, reset_model


class _StubPrediction:
    """Minimal stand-in for a surprise Prediction object."""

    def __init__(self, est: float) -> None:
        self.est = est


class _StubAlgorithm:
    """Pickle-able algorithm returning a fixed estimate, for isolation tests."""

    def __init__(self, est: float = 3.7) -> None:
        self.est = est

    def predict(self, uid: str, iid: str) -> _StubPrediction:
        return _StubPrediction(self.est)


@pytest.fixture
def stub_model_path(tmp_path: Path) -> Path:
    """Write a pickled stub algorithm to disk and return its path."""
    path = tmp_path / "stub_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(_StubAlgorithm(), f)
    return path


class TestMovieRatingModel:
    """Unit tests for MovieRatingModel class."""

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, trained_model: MovieRatingModel) -> None:
        """Test that model loads without errors."""
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model: MovieRatingModel) -> None:
        """Test that model instance has the model attribute."""
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    # =========================================================================
    # TODO 1: Prediction Return Type Tests
    # =========================================================================

    def test_predict_returns_float(self, trained_model: MovieRatingModel) -> None:
        """predict() returns a plain float, not a numpy scalar or a Prediction."""
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)
        assert not isinstance(result, bool)

    def test_predict_is_rounded_to_two_decimals(self, trained_model: MovieRatingModel) -> None:
        """The wrapper rounds to 2 decimals before returning."""
        result = trained_model.predict("196", "242")
        assert result == round(result, 2)

    # =========================================================================
    # TODO 2: Rating Range Tests
    # =========================================================================

    def test_predict_returns_value_in_valid_range(self, trained_model: MovieRatingModel) -> None:
        """Predictions are within the 1-5 rating scale."""
        result = trained_model.predict("196", "242")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_multiple_pairs_all_in_range(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Every known pair predicts inside the valid range."""
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert MIN_RATING <= result <= MAX_RATING, f"{pair} produced {result}"

    def test_predict_clips_estimate_above_maximum(self, stub_model_path: Path) -> None:
        """An estimate above 5.0 coming out of the algorithm is clipped to 5.0."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        model.model = _StubAlgorithm(est=9.9)
        assert model.predict("1", "1") == MAX_RATING

    def test_predict_clips_estimate_below_minimum(self, stub_model_path: Path) -> None:
        """An estimate below 1.0 coming out of the algorithm is clipped to 1.0."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        model.model = _StubAlgorithm(est=-4.2)
        assert model.predict("1", "1") == MIN_RATING

    # =========================================================================
    # TODO 3: Batch Prediction Tests
    # =========================================================================

    def test_predict_batch_returns_list(self, trained_model: MovieRatingModel) -> None:
        """predict_batch() returns a list."""
        results = trained_model.predict_batch([("196", "242"), ("186", "302")])
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model: MovieRatingModel) -> None:
        """predict_batch() returns one result per input pair."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model: MovieRatingModel) -> None:
        """Every batch prediction is a float inside the valid range."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("244", "51")]
        for value in trained_model.predict_batch(pairs):
            assert isinstance(value, float)
            assert MIN_RATING <= value <= MAX_RATING

    def test_predict_batch_empty_list_returns_empty_list(
        self, trained_model: MovieRatingModel
    ) -> None:
        """An empty batch is a no-op, not an error."""
        assert trained_model.predict_batch([]) == []

    def test_predict_batch_preserves_input_order(self, trained_model: MovieRatingModel) -> None:
        """Result i corresponds to input pair i."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        batch = trained_model.predict_batch(pairs)
        individual = [trained_model.predict(u, m) for u, m in pairs]
        assert batch == individual

    # =========================================================================
    # TODO 4: is_loaded() Tests
    # =========================================================================

    def test_is_loaded_returns_bool(self, trained_model: MovieRatingModel) -> None:
        """is_loaded() returns an actual bool."""
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model: MovieRatingModel) -> None:
        """is_loaded() is True once the artifact is in memory."""
        assert trained_model.is_loaded() is True

    def test_is_loaded_returns_false_when_model_cleared(self, stub_model_path: Path) -> None:
        """is_loaded() is False when the underlying algorithm is gone."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        model.model = None
        assert model.is_loaded() is False

    # =========================================================================
    # TODO 5: Error Handling Tests
    # =========================================================================

    def test_predict_raises_when_model_not_loaded(self, stub_model_path: Path) -> None:
        """predict() refuses to run without a loaded algorithm."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        model.model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict("196", "242")

    def test_predict_batch_raises_when_model_not_loaded(self, stub_model_path: Path) -> None:
        """predict_batch() refuses to run without a loaded algorithm."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        model.model = None
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict_batch([("196", "242")])

    def test_predict_with_none_user_id(self, trained_model: MovieRatingModel) -> None:
        """
        A None user_id must not corrupt the response contract.

        The model may treat it as an unknown user (cold start) or raise - both
        are acceptable - but it must never return an out-of-range rating.
        """
        try:
            result = trained_model.predict(None, "242")  # type: ignore[arg-type]
        except (TypeError, ValueError, AttributeError, KeyError):
            return
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_with_empty_string(self, trained_model: MovieRatingModel) -> None:
        """Empty IDs behave like unknown IDs: a valid cold-start rating or an error."""
        try:
            result = trained_model.predict("", "")
        except (TypeError, ValueError, AttributeError, KeyError):
            return
        assert MIN_RATING <= result <= MAX_RATING


class TestModelFileHandling:
    """Tests for model file handling."""

    def test_model_raises_error_for_missing_file(self) -> None:
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")

    def test_model_raises_error_for_corrupted_file(self, tmp_path: Path) -> None:
        """A file that is not a valid pickle surfaces as an error, not a silent None."""
        corrupted = tmp_path / "corrupted.pkl"
        corrupted.write_bytes(b"this is definitely not a pickle")
        with pytest.raises(Exception):
            MovieRatingModel(model_path=str(corrupted))

    def test_model_path_is_stored(self, stub_model_path: Path) -> None:
        """The wrapper remembers where it loaded the artifact from."""
        model = MovieRatingModel(model_path=str(stub_model_path))
        assert model.model_path == str(stub_model_path)


class TestColdStartDetection:
    """
    is_known_user / is_known_movie, ported from Lab 1.

    Cold start is not an error, but it *is* a different regime. Being able to
    tell the two apart turns "the prediction looked odd" into a fact.
    """

    def test_known_user_is_recognised(self, trained_model: MovieRatingModel) -> None:
        """User 196 rated movies in MovieLens 100K, so the model has seen them."""
        assert trained_model.is_known_user("196") is True

    def test_known_movie_is_recognised(self, trained_model: MovieRatingModel) -> None:
        """Movie 242 is in the training set."""
        assert trained_model.is_known_movie("242") is True

    def test_unknown_user_is_reported(self, trained_model: MovieRatingModel) -> None:
        """An ID that was never trained on is reported as unknown, not guessed at."""
        assert trained_model.is_known_user("no_such_user_12345") is False

    def test_unknown_movie_is_reported(self, trained_model: MovieRatingModel) -> None:
        """Same on the item side."""
        assert trained_model.is_known_movie("no_such_movie_12345") is False

    def test_zero_padded_id_is_not_the_same_user(self, trained_model: MovieRatingModel) -> None:
        """
        '0196' is NOT '196'.

        Previously this could only be inferred from a differing prediction; now
        it is a direct assertion.
        """
        assert trained_model.is_known_user("196") is True
        assert trained_model.is_known_user("0196") is False

    def test_cold_start_flags_agree_with_prediction_fallback(
        self, trained_model: MovieRatingModel
    ) -> None:
        """An unknown pair still predicts, and still predicts in range."""
        assert trained_model.is_known_user("brand_new_user") is False
        assert MIN_RATING <= trained_model.predict("brand_new_user", "242") <= MAX_RATING

    def test_cold_start_checks_are_safe_without_a_trainset(self, stub_model_path: Path) -> None:
        """
        A stub algorithm exposes no trainset; the check reports unknown rather
        than raising AttributeError.
        """
        model = MovieRatingModel(model_path=str(stub_model_path))
        assert model.is_known_user("196") is False
        assert model.is_known_movie("242") is False


class TestBlankIdRejection:
    """predict() rejects blank IDs before touching the algorithm (from Lab 1)."""

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_user_id_raises_value_error(
        self, trained_model: MovieRatingModel, blank: str
    ) -> None:
        """A blank ID is a caller mistake, surfaced as ValueError -> HTTP 422."""
        with pytest.raises(ValueError, match="non-empty"):
            trained_model.predict(blank, "242")

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_movie_id_raises_value_error(
        self, trained_model: MovieRatingModel, blank: str
    ) -> None:
        """The same rule applies to the item side."""
        with pytest.raises(ValueError, match="non-empty"):
            trained_model.predict("196", blank)

    def test_batch_propagates_blank_id_error(self, trained_model: MovieRatingModel) -> None:
        """One bad pair fails the batch rather than silently returning garbage."""
        with pytest.raises(ValueError):
            trained_model.predict_batch([("196", "242"), ("", "302")])


class TestModelLoadValidation:
    """_load_model() verifies what came out of the pickle (from Lab 1)."""

    def test_pickle_without_predict_method_is_rejected(self, tmp_path: Path) -> None:
        """
        A pickle can contain anything.

        Failing at load time turns a production 500 on the first request into a
        startup error the health check reports immediately.
        """
        path = tmp_path / "not_a_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"just": "a dict"}, f)

        with pytest.raises(ValueError, match="no predict"):
            MovieRatingModel(model_path=str(path))

    def test_corrupted_pickle_raises_value_error(self, tmp_path: Path) -> None:
        """Unpickling garbage is reported as ValueError, not a raw UnpicklingError."""
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"\x80\x04 definitely not a pickle")
        with pytest.raises(ValueError, match="Could not unpickle"):
            MovieRatingModel(model_path=str(path))

    def test_model_not_loaded_error_is_a_runtime_error(self) -> None:
        """
        ModelNotLoadedError subclasses RuntimeError.

        Existing `except RuntimeError` callers keep working, while the API layer
        can map this one case to 503 instead of a generic 500.
        """
        assert issubclass(ModelNotLoadedError, RuntimeError)


class TestModelSingleton:
    """Tests for the get_model()/reset_model() singleton helpers."""

    def test_get_model_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_model() caches one instance instead of re-reading the pickle."""
        reset_model()
        monkeypatch.setattr("app.model.MovieRatingModel.__init__", lambda self, *a, **k: None)
        first = get_model()
        second = get_model()
        assert first is second
        reset_model()

    def test_reset_model_clears_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """reset_model() forces the next get_model() to build a fresh instance."""
        reset_model()
        monkeypatch.setattr("app.model.MovieRatingModel.__init__", lambda self, *a, **k: None)
        first = get_model()
        reset_model()
        second = get_model()
        assert first is not second
        reset_model()


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
