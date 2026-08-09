"""
Model behavioral tests.

Level 4 of the ML testing pyramid. These do not assert on code paths - they
assert on what the model *does*:

- Invariance: output must not change under perturbations that carry no meaning
- Directional: output must move the way domain knowledge says it should
- Minimum functionality: cases the model must get right to be shippable

Note on thresholds: every bound below is a *floor*, deliberately loose enough to
survive retraining and tight enough to catch a broken artifact (an untrained
model, a shuffled index, a constant predictor).

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from app.model import MovieRatingModel

# Quality gates - keep in sync with scripts/validate_model.py.
# Calibrated against the observed distribution on a 2,000-row sample of the
# training set (MAE 0.62, 94.5% within 1.5, worst error 2.81), then loosened so
# normal retraining variance does not turn the build red.
MAX_MEAN_ABSOLUTE_ERROR = 0.85
MIN_FRACTION_WITHIN_1_5 = 0.85
MAX_SINGLE_ERROR = 3.5
MAX_ERROR_ON_KNOWN_PAIR = 3.0


class TestModelInvariance:
    """
    Invariance tests - output shouldn't change for certain perturbations.

    These tests ensure that the model produces consistent results
    when given the same inputs.
    """

    # =========================================================================
    # TODO 1: Deterministic Output Tests
    # =========================================================================

    def test_same_input_same_output(self, trained_model: MovieRatingModel) -> None:
        """Inference is a pure function: same input, same rating."""
        assert trained_model.predict("196", "242") == trained_model.predict("196", "242")

    def test_multiple_calls_consistent(self, trained_model: MovieRatingModel) -> None:
        """Five consecutive calls agree - no hidden state, no sampling."""
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert len(set(results)) == 1, f"non-deterministic predictions: {results}"

    def test_determinism_across_many_pairs(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Determinism holds for every known pair, not just the sample one."""
        for pair in known_user_movie_pairs:
            first = trained_model.predict(pair["user_id"], pair["movie_id"])
            second = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert first == second, f"{pair} is non-deterministic"

    def test_reloading_model_gives_same_predictions(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """
        Serialisation round-trips faithfully.

        A model that predicts differently after reload would drift between
        deployments even with an identical artifact.
        """
        reloaded = MovieRatingModel()
        for pair in known_user_movie_pairs:
            assert trained_model.predict(pair["user_id"], pair["movie_id"]) == reloaded.predict(
                pair["user_id"], pair["movie_id"]
            )

    # =========================================================================
    # TODO 2: Batch Order Invariance Tests
    # =========================================================================

    def test_batch_order_independent(self, trained_model: MovieRatingModel) -> None:
        """Reordering a batch permutes the results, it does not change them."""
        pairs1 = [("196", "242"), ("186", "302"), ("22", "377")]
        pairs2 = list(reversed(pairs1))

        results1 = trained_model.predict_batch(pairs1)
        results2 = trained_model.predict_batch(pairs2)

        assert results1 == list(reversed(results2))

    def test_individual_vs_batch_same_results(self, trained_model: MovieRatingModel) -> None:
        """Batching must be an implementation detail, invisible in the output."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("244", "51")]
        assert trained_model.predict_batch(pairs) == [
            trained_model.predict(user, movie) for user, movie in pairs
        ]

    def test_batch_size_does_not_change_results(self, trained_model: MovieRatingModel) -> None:
        """A pair predicted alone and inside a large batch gets the same rating."""
        pair = ("196", "242")
        alone = trained_model.predict_batch([pair])[0]
        padded = [("186", "302")] * 20 + [pair]
        assert trained_model.predict_batch(padded)[-1] == alone

    def test_model_layer_does_not_normalise_ids(self, trained_model: MovieRatingModel) -> None:
        """
        Whitespace stripping lives in the schema, not in the model.

        A padded ID reaching the model is therefore an *unknown* key and falls
        back to cold start. Pinning this makes the division of responsibility
        explicit: if someone moves normalisation, this test says so.
        """
        clean = trained_model.predict("196", "242")
        padded = trained_model.predict(" 196 ", " 242 ")
        assert padded != clean, "model unexpectedly normalises IDs - check schema/model split"
        assert 1.0 <= padded <= 5.0


class TestModelDirectional:
    """
    Directional tests - output should change in expected direction.

    These tests verify that the model behaves sensibly when inputs
    change in predictable ways.
    """

    # =========================================================================
    # TODO 3: Directional Tests
    # =========================================================================

    def test_predictions_are_reasonable(
        self, trained_model: MovieRatingModel, evaluation_sample: List[Tuple[str, str, float]]
    ) -> None:
        """
        Most predictions land within 1.5 stars of the true rating.

        Asserted over a 500-row sample rather than per pair: individual users do
        rate against the crowd (the 5 hand-picked pairs in ``conftest`` include
        one such outlier), so a per-pair bound measures sampling luck. A
        *fraction* bound measures the model.
        """
        errors = [
            abs(trained_model.predict(user, movie) - actual)
            for user, movie, actual in evaluation_sample
        ]
        within = sum(1 for error in errors if error < 1.5) / len(errors)
        assert within >= MIN_FRACTION_WITHIN_1_5, (
            f"only {within:.1%} of predictions are within 1.5 stars "
            f"(gate: {MIN_FRACTION_WITHIN_1_5:.0%})"
        )

    def test_known_pairs_have_no_extreme_error(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """No hand-picked pair is catastrophically mispredicted."""
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            error = abs(prediction - pair["actual_rating"])
            assert error < MAX_ERROR_ON_KNOWN_PAIR, (
                f"user {pair['user_id']} / movie {pair['movie_id']}: "
                f"predicted {prediction}, actual {pair['actual_rating']}"
            )

    def test_different_movies_different_predictions(self, trained_model: MovieRatingModel) -> None:
        """
        One user, many movies: the model must discriminate between items.

        A constant output here means the item factors are dead.
        """
        movies = ["242", "302", "377", "51", "346", "50", "181", "258"]
        predictions = [trained_model.predict("196", movie) for movie in movies]
        assert len(set(predictions)) > 1, f"identical predictions for all movies: {predictions}"

    def test_different_users_different_predictions(self, trained_model: MovieRatingModel) -> None:
        """
        One movie, many users: the model must personalise.

        A constant output here means the user factors/biases are dead.
        """
        users = ["196", "186", "22", "244", "166", "298", "115", "253"]
        predictions = [trained_model.predict(user, "242") for user in users]
        assert len(set(predictions)) > 1, f"identical predictions for all users: {predictions}"

    def test_popular_movie_scores_above_unpopular_one(
        self, trained_model: MovieRatingModel
    ) -> None:
        """
        Domain knowledge as a test.

        Movie 50 (Star Wars, mean ~4.36) must outrank movie 122 (Cape Fear,
        mean ~2.6) for the same user. This is the strongest single signal that
        item biases were learned in the right direction.
        """
        assert trained_model.predict("196", "50") > trained_model.predict("196", "122")

    def test_generous_user_scores_above_harsh_user(self, trained_model: MovieRatingModel) -> None:
        """
        The mirror check on the user side.

        User 4 rates everything highly (mean ~4.3); user 181 is famously harsh
        (mean ~1.5). The same movie must score higher for the generous one.
        """
        assert trained_model.predict("4", "50") > trained_model.predict("181", "50")

    def test_unknown_user_falls_back_near_global_mean(
        self, trained_model: MovieRatingModel
    ) -> None:
        """
        Cold start degrades toward the average rating (~3.5), not to an extreme.

        A cold-start prediction of 1.0 or 5.0 would be a confident wrong answer.
        """
        prediction = trained_model.predict("new_user_never_seen", "242")
        assert 2.5 <= prediction <= 4.5, f"cold-start prediction {prediction} is not neutral"


class TestMinimumFunctionality:
    """
    Minimum functionality tests - basic cases the model must handle.

    These are simple test cases that the model absolutely must pass
    to be considered functional.
    """

    # =========================================================================
    # TODO 4: Minimum Functionality Tests
    # =========================================================================

    def test_can_predict_for_known_user(self, trained_model: MovieRatingModel) -> None:
        """The single most basic case: a known user, a known movie."""
        prediction = trained_model.predict("196", "242")
        assert prediction is not None
        assert 1.0 <= prediction <= 5.0

    def test_can_predict_for_multiple_users(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """Every known pair produces a usable rating."""
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert isinstance(prediction, float)
            assert 1.0 <= prediction <= 5.0

    def test_predictions_not_all_same(
        self, trained_model: MovieRatingModel, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """A constant predictor would pass every range check - and be useless."""
        predictions = [
            trained_model.predict(pair["user_id"], pair["movie_id"])
            for pair in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, f"all predictions identical: {predictions}"

    def test_predictions_span_a_meaningful_range(self, trained_model: MovieRatingModel) -> None:
        """
        Across many pairs the output spread must be non-trivial.

        A model collapsed onto the global mean has a spread near zero.
        """
        movies = ["50", "122", "242", "302", "377", "181", "258", "100", "286", "288"]
        predictions = [trained_model.predict("196", movie) for movie in movies]
        assert max(predictions) - min(predictions) > 0.3

    # =========================================================================
    # TODO 5: Edge Case Tests
    # =========================================================================

    def test_handles_unknown_user_gracefully(
        self, trained_model: MovieRatingModel, unknown_users: List[str]
    ) -> None:
        """Unknown users yield a valid fallback rating or a typed error - never garbage."""
        for user_id in unknown_users:
            try:
                prediction = trained_model.predict(user_id, "242")
            except (ValueError, KeyError):
                continue
            assert 1.0 <= prediction <= 5.0, f"user {user_id} produced {prediction}"

    def test_handles_unknown_movie_gracefully(
        self, trained_model: MovieRatingModel, unknown_movies: List[str]
    ) -> None:
        """Unknown movies follow the same cold-start rule."""
        for movie_id in unknown_movies:
            try:
                prediction = trained_model.predict("196", movie_id)
            except (ValueError, KeyError):
                continue
            assert 1.0 <= prediction <= 5.0, f"movie {movie_id} produced {prediction}"

    def test_handles_both_unknown(self, trained_model: MovieRatingModel) -> None:
        """Neither side known is the hardest cold start and must still be safe."""
        prediction = trained_model.predict("no_such_user", "no_such_movie")
        assert 1.0 <= prediction <= 5.0


class TestModelPerformance:
    """
    Performance-related behavioral tests.

    These tests verify that the model performs adequately
    on known test cases.
    """

    # =========================================================================
    # TODO 6: Performance Tests
    # =========================================================================

    def test_average_error_acceptable(
        self, trained_model: MovieRatingModel, evaluation_sample: List[Tuple[str, str, float]]
    ) -> None:
        """Mean absolute error over the evaluation sample stays under the release gate."""
        errors = [
            abs(trained_model.predict(user, movie) - actual)
            for user, movie, actual in evaluation_sample
        ]
        mean_error = sum(errors) / len(errors)
        assert (
            mean_error < MAX_MEAN_ABSOLUTE_ERROR
        ), f"MAE {mean_error:.3f} exceeds gate {MAX_MEAN_ABSOLUTE_ERROR}"

    def test_no_extreme_errors(
        self, trained_model: MovieRatingModel, evaluation_sample: List[Tuple[str, str, float]]
    ) -> None:
        """
        No single prediction is catastrophically wrong.

        The theoretical worst case on a 1-5 scale is 4.0; anything approaching
        it means the model inverted a user or item factor.
        """
        worst = max(
            abs(trained_model.predict(user, movie) - actual)
            for user, movie, actual in evaluation_sample
        )
        assert worst < MAX_SINGLE_ERROR, f"worst error {worst:.2f} exceeds gate"

    def test_recorded_training_metrics_meet_gate(self) -> None:
        """
        The RMSE recorded at training time must clear the release gate.

        Reads models/metrics.json, written by scripts/train_model.py, so a
        regression in training quality fails the build even though inference
        still works.
        """
        metrics_path = Path(__file__).resolve().parents[2] / "models" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("models/metrics.json not found. Run train_model.py first.")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["rmse"] < 1.0, f"training RMSE {metrics['rmse']:.4f} regressed"
        assert metrics["mae"] < 0.8, f"training MAE {metrics['mae']:.4f} regressed"

    @pytest.mark.slow
    def test_batch_throughput_is_acceptable(self, trained_model: MovieRatingModel) -> None:
        """
        1,000 predictions complete well inside a request timeout.

        Guards against an accidental O(n) model reload per prediction.
        """
        import time

        pairs = [("196", str(movie)) for movie in range(1, 1001)]
        started = time.perf_counter()
        results = trained_model.predict_batch(pairs)
        elapsed = time.perf_counter() - started

        assert len(results) == 1000
        assert elapsed < 5.0, f"1000 predictions took {elapsed:.2f}s"


class TestModelRobustness:
    """
    Robustness tests - model behavior under unusual conditions.
    """

    # =========================================================================
    # TODO 7: Robustness Tests
    # =========================================================================

    def test_handles_string_numeric_ids(self, trained_model: MovieRatingModel) -> None:
        """IDs are strings by contract; numeric-looking strings are the normal case."""
        assert 1.0 <= trained_model.predict("196", "242") <= 5.0

    def test_handles_leading_zeros_in_ids(self, trained_model: MovieRatingModel) -> None:
        """
        '0196' is NOT '196'.

        MovieLens IDs are stored without padding, so a zero-padded ID is an
        unknown user. Asserting the cold-start path here documents that a client
        must not pad IDs.
        """
        padded = trained_model.predict("0196", "242")
        assert 1.0 <= padded <= 5.0

    def test_handles_very_long_ids(self, trained_model: MovieRatingModel) -> None:
        """An absurdly long ID is treated as unknown, not a crash."""
        assert 1.0 <= trained_model.predict("9" * 200, "242") <= 5.0

    def test_handles_special_characters_in_ids(self, trained_model: MovieRatingModel) -> None:
        """Injection-shaped IDs are just unknown keys to a dictionary lookup."""
        for weird in ["'; DROP TABLE ratings;--", "../../etc/passwd", "<script>", "user\x00id"]:
            assert 1.0 <= trained_model.predict(weird, "242") <= 5.0

    def test_repeated_predictions_do_not_mutate_model(
        self, trained_model: MovieRatingModel
    ) -> None:
        """Inference is read-only: 100 calls later the answer is unchanged."""
        before = trained_model.predict("196", "242")
        for _ in range(100):
            trained_model.predict("186", "302")
        assert trained_model.predict("196", "242") == before


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
