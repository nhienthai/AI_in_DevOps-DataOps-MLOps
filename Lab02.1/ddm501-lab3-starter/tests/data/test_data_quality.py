"""
Data quality tests.

Level 3 of the ML testing pyramid. Two halves, both necessary:

1. The validators themselves are tested against clean *and* deliberately
   corrupted data - a validator that never fires is indistinguishable from one
   that does not work.
2. The real MovieLens 100K dataset is validated against the contract the
   training pipeline assumes (schema, completeness, range, uniqueness, volume,
   distribution). These are skipped when the dataset is not cached locally.

Run tests:
    pytest tests/data/test_data_quality.py -v
"""

from typing import Any, Dict, List

import pandas as pd
import pytest

from training import dataset
from training.validators import (
    ValidationReport,
    validate_completeness,
    validate_dtypes,
    validate_mean_within,
    validate_min_rows,
    validate_no_duplicates,
    validate_ratings_dataset,
    validate_schema,
    validate_value_range,
)

REQUIRED_COLUMNS = ["user_id", "movie_id", "rating"]


# =============================================================================
# Part 1 - the validators (fast, no dataset needed)
# =============================================================================


class TestValidationReport:
    """Tests for the report object the validators return."""

    def test_empty_report_is_valid(self) -> None:
        """A report with no errors is valid."""
        assert ValidationReport().is_valid is True

    def test_report_with_error_is_invalid(self) -> None:
        """Recording an error flips the report to invalid."""
        report = ValidationReport()
        report.add("something is wrong")
        assert report.is_valid is False

    def test_raise_if_invalid_raises_with_all_messages(self) -> None:
        """The pipeline gate reports every failure at once, not just the first."""
        report = ValidationReport()
        report.add("first problem")
        report.add("second problem")
        with pytest.raises(ValueError) as exc_info:
            report.raise_if_invalid()
        assert "first problem" in str(exc_info.value)
        assert "second problem" in str(exc_info.value)

    def test_raise_if_invalid_is_silent_when_valid(self) -> None:
        """A clean report does not block the pipeline."""
        ValidationReport().raise_if_invalid()


class TestSchemaValidation:
    """3.2 - Schema validation."""

    def test_valid_schema_passes(self, ratings_dataframe: pd.DataFrame) -> None:
        """The sample data has the columns the pipeline expects."""
        assert validate_schema(ratings_dataframe, REQUIRED_COLUMNS).is_valid

    def test_missing_column_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """Dropping a required column fails validation."""
        report = validate_schema(ratings_dataframe.drop(columns=["rating"]), REQUIRED_COLUMNS)
        assert not report.is_valid
        assert "rating" in report.errors[0]

    def test_extra_columns_are_allowed(self, ratings_dataframe: pd.DataFrame) -> None:
        """Additional columns (e.g. timestamp) do not break the contract."""
        frame = ratings_dataframe.assign(timestamp=0)
        assert validate_schema(frame, REQUIRED_COLUMNS).is_valid

    def test_correct_dtypes_pass(self, ratings_dataframe: pd.DataFrame) -> None:
        """IDs are strings (object), ratings are floats."""
        report = validate_dtypes(
            ratings_dataframe, {"user_id": "O", "movie_id": "O", "rating": "f"}
        )
        assert report.is_valid, report.errors

    def test_wrong_dtype_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """A rating column stored as text is a schema violation."""
        frame = ratings_dataframe.assign(rating=ratings_dataframe["rating"].astype(str))
        report = validate_dtypes(frame, {"rating": "f"})
        assert not report.is_valid

    def test_dtype_check_on_missing_column_is_reported(self) -> None:
        """Checking a column that is not there is an error, not a crash."""
        report = validate_dtypes(pd.DataFrame({"a": [1]}), {"rating": "f"})
        assert not report.is_valid


class TestCompleteness:
    """Completeness - no unexpected missing values."""

    def test_complete_data_passes(self, ratings_dataframe: pd.DataFrame) -> None:
        """The sample data has no gaps."""
        assert validate_completeness(ratings_dataframe, REQUIRED_COLUMNS).is_valid

    def test_null_rating_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """A null rating cannot be trained on and must be caught."""
        frame = ratings_dataframe.copy()
        frame.loc[0, "rating"] = None
        report = validate_completeness(frame, REQUIRED_COLUMNS)
        assert not report.is_valid
        assert "rating" in report.errors[0]

    def test_blank_user_id_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """An empty string ID passes a null check but is still unusable."""
        frame = ratings_dataframe.copy()
        frame.loc[0, "user_id"] = "  "
        report = validate_completeness(frame, REQUIRED_COLUMNS)
        assert not report.is_valid

    def test_completeness_on_missing_column_is_reported(self) -> None:
        """Missing columns are reported rather than silently skipped."""
        report = validate_completeness(pd.DataFrame({"a": [1]}), ["rating"])
        assert not report.is_valid


class TestValueRange:
    """Distribution - values inside the expected domain."""

    def test_ratings_in_valid_range(self, sample_ratings: List[Dict[str, Any]]) -> None:
        """Every sample rating sits on the 1-5 scale."""
        for row in sample_ratings:
            assert 1.0 <= row["rating"] <= 5.0

    def test_valid_range_passes(self, ratings_dataframe: pd.DataFrame) -> None:
        """The range validator agrees with the manual check above."""
        assert validate_value_range(ratings_dataframe, "rating", 1.0, 5.0).is_valid

    @pytest.mark.parametrize("bad_rating", [0.0, 0.9, 5.1, 10.0, -3.0])
    def test_out_of_range_rating_is_detected(
        self, ratings_dataframe: pd.DataFrame, bad_rating: float
    ) -> None:
        """Any value off the scale is caught, on either side."""
        frame = ratings_dataframe.copy()
        frame.loc[0, "rating"] = bad_rating
        assert not validate_value_range(frame, "rating", 1.0, 5.0).is_valid

    def test_boundary_ratings_are_accepted(self, ratings_dataframe: pd.DataFrame) -> None:
        """1.0 and 5.0 are valid ratings, not off-by-one failures."""
        frame = ratings_dataframe.copy()
        frame.loc[0, "rating"] = 1.0
        frame.loc[1, "rating"] = 5.0
        assert validate_value_range(frame, "rating", 1.0, 5.0).is_valid

    def test_non_numeric_rating_is_detected(self) -> None:
        """A rating of 'four' is a type failure the range check surfaces."""
        frame = pd.DataFrame({"rating": [4.0, "four"]})
        assert not validate_value_range(frame, "rating", 1.0, 5.0).is_valid

    def test_range_check_on_missing_column_is_reported(self) -> None:
        """A missing column is an error, not a pass."""
        assert not validate_value_range(pd.DataFrame({"a": [1]}), "rating", 1.0, 5.0).is_valid


class TestUniqueness:
    """Uniqueness - one rating per user-movie pair."""

    def test_unique_pairs_pass(self, ratings_dataframe: pd.DataFrame) -> None:
        """The sample data has no repeated user-movie pair."""
        assert validate_no_duplicates(ratings_dataframe, ["user_id", "movie_id"]).is_valid

    def test_duplicate_pair_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """A repeated pair would double-weight that observation during training."""
        frame = pd.concat([ratings_dataframe, ratings_dataframe.head(1)], ignore_index=True)
        report = validate_no_duplicates(frame, ["user_id", "movie_id"])
        assert not report.is_valid
        assert "1 duplicate" in report.errors[0]

    def test_duplicate_check_on_missing_column_is_reported(self) -> None:
        """The check reports missing key columns instead of raising KeyError."""
        report = validate_no_duplicates(pd.DataFrame({"a": [1]}), ["user_id", "movie_id"])
        assert not report.is_valid


class TestVolumeAndDistribution:
    """Volume and distribution checks used as drift guards."""

    def test_min_rows_passes(self, ratings_dataframe: pd.DataFrame) -> None:
        """The sample set clears a small volume floor."""
        assert validate_min_rows(ratings_dataframe, 5).is_valid

    def test_truncated_dataset_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """Silent truncation - the classic pipeline bug - is caught by row count."""
        assert not validate_min_rows(ratings_dataframe.head(2), 5).is_valid

    def test_mean_within_band_passes(self, ratings_dataframe: pd.DataFrame) -> None:
        """The sample mean sits in a plausible band."""
        assert validate_mean_within(ratings_dataframe, "rating", 1.0, 5.0).is_valid

    def test_shifted_mean_is_detected(self, ratings_dataframe: pd.DataFrame) -> None:
        """A distribution shift moves the mean out of its historical band."""
        frame = ratings_dataframe.assign(rating=5.0)
        assert not validate_mean_within(frame, "rating", 3.0, 4.0).is_valid

    def test_mean_check_on_missing_column_is_reported(self) -> None:
        """A missing column is an error, not a pass."""
        assert not validate_mean_within(pd.DataFrame({"a": [1]}), "rating", 1.0, 5.0).is_valid


class TestFullContract:
    """The combined gate used before training."""

    def test_clean_dataset_passes_full_contract(self, ratings_dataframe: pd.DataFrame) -> None:
        """Clean data passes every check at once."""
        report = validate_ratings_dataset(ratings_dataframe)
        assert report.is_valid, report.errors

    def test_corrupted_dataset_reports_multiple_failures(
        self, corrupted_ratings: List[Dict[str, Any]]
    ) -> None:
        """
        The deliberately broken fixture trips several checks.

        This is the test that proves the suite has teeth: if this ever passes,
        the validators have stopped working.
        """
        report = validate_ratings_dataset(pd.DataFrame(corrupted_ratings))
        assert not report.is_valid
        assert len(report.errors) >= 2

    def test_missing_column_short_circuits(self, ratings_dataframe: pd.DataFrame) -> None:
        """Without the schema there is nothing else worth checking."""
        report = validate_ratings_dataset(ratings_dataframe.drop(columns=["rating"]))
        assert not report.is_valid
        assert len(report.errors) == 1


# =============================================================================
# Part 2 - the real MovieLens 100K dataset
# =============================================================================


class TestMovieLensDataset:
    """Validate the actual training data against the documented contract."""

    def test_dataset_has_expected_schema(self, movielens_ratings: pd.DataFrame) -> None:
        """The raw file parses into the four documented columns."""
        assert list(movielens_ratings.columns) == dataset.RATING_COLUMNS

    def test_dataset_has_expected_row_count(self, movielens_ratings: pd.DataFrame) -> None:
        """MovieLens 100K must contain exactly 100,000 ratings."""
        assert len(movielens_ratings) == dataset.EXPECTED_N_RATINGS

    def test_dataset_has_expected_user_count(self, movielens_ratings: pd.DataFrame) -> None:
        """943 distinct users, as documented by GroupLens."""
        assert movielens_ratings["user_id"].nunique() == dataset.EXPECTED_N_USERS

    def test_dataset_has_expected_movie_count(self, movielens_ratings: pd.DataFrame) -> None:
        """1,682 distinct movies, as documented by GroupLens."""
        assert movielens_ratings["movie_id"].nunique() == dataset.EXPECTED_N_MOVIES

    def test_dataset_has_no_missing_values(self, movielens_ratings: pd.DataFrame) -> None:
        """No gaps anywhere in the training data."""
        report = validate_completeness(movielens_ratings, REQUIRED_COLUMNS)
        assert report.is_valid, report.errors

    def test_dataset_ratings_in_valid_range(self, movielens_ratings: pd.DataFrame) -> None:
        """Every rating is on the 1-5 scale the API promises to return."""
        report = validate_value_range(movielens_ratings, "rating", 1.0, 5.0)
        assert report.is_valid, report.errors

    def test_dataset_ratings_are_whole_numbers(self, movielens_ratings: pd.DataFrame) -> None:
        """MovieLens 100K uses integer stars only - half stars would be new data."""
        assert (movielens_ratings["rating"] % 1 == 0).all()

    def test_dataset_has_no_duplicate_pairs(self, movielens_ratings: pd.DataFrame) -> None:
        """Each user rated each movie at most once."""
        report = validate_no_duplicates(movielens_ratings, ["user_id", "movie_id"])
        assert report.is_valid, report.errors

    def test_dataset_mean_rating_is_stable(self, movielens_ratings: pd.DataFrame) -> None:
        """
        The historical mean rating is ~3.53.

        A band of [3.4, 3.7] catches a swapped or resampled dataset without
        flagging normal variation.
        """
        report = validate_mean_within(movielens_ratings, "rating", 3.4, 3.7)
        assert report.is_valid, report.errors

    def test_dataset_every_user_has_minimum_ratings(self, movielens_ratings: pd.DataFrame) -> None:
        """
        Collaborative filtering needs signal per user; MovieLens 100K guarantees
        at least 20 ratings per user. Fewer would mean a filtered dataset.
        """
        assert movielens_ratings.groupby("user_id").size().min() >= 20

    def test_dataset_ids_are_strings(self, movielens_ratings: pd.DataFrame) -> None:
        """IDs stay strings end to end, matching the API request schema."""
        report = validate_dtypes(movielens_ratings, {"user_id": "O", "movie_id": "O"})
        assert report.is_valid, report.errors

    def test_dataset_passes_full_contract(self, movielens_ratings: pd.DataFrame) -> None:
        """The single gate the training pipeline would run before fitting."""
        report = validate_ratings_dataset(movielens_ratings, min_rows=dataset.EXPECTED_N_RATINGS)
        assert report.is_valid, report.errors

    def test_known_pairs_exist_in_dataset(
        self, movielens_ratings: pd.DataFrame, known_user_movie_pairs: List[Dict[str, Any]]
    ) -> None:
        """
        The pairs the behavioural tests rely on really are in the training data.

        Without this, a "prediction close to actual" test could be silently
        measuring cold-start behaviour instead.
        """
        indexed = movielens_ratings.set_index(["user_id", "movie_id"])["rating"]
        for pair in known_user_movie_pairs:
            key = (pair["user_id"], pair["movie_id"])
            assert key in indexed.index, f"{key} not in MovieLens 100K"
            assert float(indexed.loc[key]) == pair["actual_rating"]


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
