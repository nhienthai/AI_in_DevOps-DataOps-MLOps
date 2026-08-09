"""
Custom data validators for the ratings dataset.

Data tests are the level of the ML testing pyramid that traditional software
testing has no equivalent for: the code can be perfect and the system still
wrong because the data drifted, lost rows, or changed type. These validators
are deliberately plain functions returning a report, so they can be used both
from the test suite and as a pre-training gate in the pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class ValidationReport:
    """Outcome of one or more validation checks."""

    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no check reported a problem."""
        return not self.errors

    def add(self, message: str) -> None:
        """Record a validation failure."""
        self.errors.append(message)

    def extend(self, other: "ValidationReport") -> "ValidationReport":
        """Merge another report into this one and return self."""
        self.errors.extend(other.errors)
        return self

    def raise_if_invalid(self) -> None:
        """Raise a ValueError listing every failure, for use as a pipeline gate."""
        if not self.is_valid:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.errors))


def validate_schema(frame: pd.DataFrame, expected_columns: Sequence[str]) -> ValidationReport:
    """Check that every expected column is present."""
    report = ValidationReport()
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        report.add(f"Missing expected columns: {missing}")
    return report


def validate_dtypes(frame: pd.DataFrame, expected: Dict[str, str]) -> ValidationReport:
    """
    Check column dtypes against a mapping of ``column -> pandas dtype kind``.

    ``kind`` uses numpy's single-letter codes: ``O`` object/str, ``f`` float,
    ``i`` int, ``b`` bool.
    """
    report = ValidationReport()
    for column, kind in expected.items():
        if column not in frame.columns:
            report.add(f"Cannot check dtype of missing column '{column}'")
            continue
        actual = frame[column].dtype.kind
        if actual != kind:
            report.add(f"Column '{column}' has dtype kind '{actual}', expected '{kind}'")
    return report


def validate_completeness(frame: pd.DataFrame, required_columns: Sequence[str]) -> ValidationReport:
    """Check that required columns contain no nulls and no blank strings."""
    report = ValidationReport()
    for column in required_columns:
        if column not in frame.columns:
            report.add(f"Cannot check completeness of missing column '{column}'")
            continue
        n_null = int(frame[column].isna().sum())
        if n_null:
            report.add(f"Column '{column}' has {n_null} missing value(s)")
        if frame[column].dtype.kind == "O":
            # map() rather than the .str accessor: pandas-stubs mistypes
            # `.astype(str).str` and rejects the chained call, while this form
            # is equally clear and type-checks cleanly.
            n_blank = int(frame[column].map(lambda value: str(value).strip() == "").sum())
            if n_blank:
                report.add(f"Column '{column}' has {n_blank} blank value(s)")
    return report


def validate_value_range(
    frame: pd.DataFrame, column: str, minimum: float, maximum: float
) -> ValidationReport:
    """Check that a numeric column stays inside ``[minimum, maximum]``."""
    report = ValidationReport()
    if column not in frame.columns:
        report.add(f"Cannot check range of missing column '{column}'")
        return report

    values = pd.to_numeric(frame[column], errors="coerce")
    n_non_numeric = int(values.isna().sum() - frame[column].isna().sum())
    if n_non_numeric > 0:
        report.add(f"Column '{column}' has {n_non_numeric} non-numeric value(s)")

    out_of_range = values.dropna()
    n_out = int(((out_of_range < minimum) | (out_of_range > maximum)).sum())
    if n_out:
        report.add(f"Column '{column}' has {n_out} value(s) outside [{minimum}, {maximum}]")
    return report


def validate_no_duplicates(frame: pd.DataFrame, keys: Sequence[str]) -> ValidationReport:
    """Check that ``keys`` form a unique key (one rating per user-movie pair)."""
    report = ValidationReport()
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        report.add(f"Cannot check duplicates, missing columns: {missing}")
        return report

    n_duplicates = int(frame.duplicated(subset=list(keys)).sum())
    if n_duplicates:
        report.add(f"Found {n_duplicates} duplicate row(s) for key {list(keys)}")
    return report


def validate_mean_within(
    frame: pd.DataFrame, column: str, low: float, high: float
) -> ValidationReport:
    """
    Check that a column's mean sits inside an expected band.

    This is the cheapest useful drift detector: a training set whose mean rating
    suddenly moves outside the historical band is a signal to stop, not retrain.
    """
    report = ValidationReport()
    if column not in frame.columns:
        report.add(f"Cannot check mean of missing column '{column}'")
        return report

    mean = float(pd.to_numeric(frame[column], errors="coerce").mean())
    if not low <= mean <= high:
        report.add(f"Mean of '{column}' is {mean:.4f}, expected within [{low}, {high}]")
    return report


def validate_min_rows(frame: pd.DataFrame, minimum: int) -> ValidationReport:
    """Check that the dataset is not unexpectedly small (silent truncation)."""
    report = ValidationReport()
    if len(frame) < minimum:
        report.add(f"Dataset has {len(frame)} rows, expected at least {minimum}")
    return report


def validate_ratings_dataset(
    frame: pd.DataFrame,
    min_rating: float = 1.0,
    max_rating: float = 5.0,
    min_rows: int = 1,
) -> ValidationReport:
    """
    Run the full ratings-data contract and return a single merged report.

    Suitable as a pre-training gate:

        validate_ratings_dataset(df).raise_if_invalid()
    """
    required = ["user_id", "movie_id", "rating"]
    report = ValidationReport()
    report.extend(validate_schema(frame, required))
    if not report.is_valid:
        return report

    checks: Iterable[ValidationReport] = (
        validate_min_rows(frame, min_rows),
        validate_completeness(frame, required),
        validate_value_range(frame, "rating", min_rating, max_rating),
        validate_no_duplicates(frame, ["user_id", "movie_id"]),
    )
    for check in checks:
        report.extend(check)
    return report
