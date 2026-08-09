"""
MovieLens 100K dataset access.

Downloads and caches the dataset in the *same* directory scikit-surprise uses
(``~/.surprise_data/ml-100k/ml-100k/u.data``), so both training backends and the
data-quality tests share one cache and one copy on disk.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]

#: Expected shape of the raw dataset, used by the data-quality tests.
EXPECTED_N_RATINGS = 100_000
EXPECTED_N_USERS = 943
EXPECTED_N_MOVIES = 1682
MIN_RATING = 1.0
MAX_RATING = 5.0


def get_dataset_dir() -> Path:
    """Return the cache directory, honouring surprise's SURPRISE_DATA_FOLDER."""
    return Path(os.getenv("SURPRISE_DATA_FOLDER", str(Path.home() / ".surprise_data")))


def get_ratings_path() -> Path:
    """Return the expected path of the raw ``u.data`` ratings file."""
    return get_dataset_dir() / "ml-100k" / "ml-100k" / "u.data"


def ensure_ml100k(force: bool = False) -> Path:
    """
    Make sure MovieLens 100K is available locally and return the ratings path.

    Args:
        force: re-download even if the file is already cached.

    Returns:
        Path to ``u.data``.
    """
    path = get_ratings_path()
    if path.exists() and not force:
        return path

    target_dir = get_dataset_dir() / "ml-100k"
    target_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(ML100K_URL, timeout=120) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target_dir)

    if not path.exists():
        raise FileNotFoundError(f"Download succeeded but {path} is missing")
    return path


def load_ratings(download: bool = True) -> pd.DataFrame:
    """
    Load the raw ratings as a DataFrame with columns ``user_id``, ``movie_id``,
    ``rating``, ``timestamp``. IDs are strings to match the serving contract.
    """
    path = ensure_ml100k() if download else get_ratings_path()
    return pd.read_csv(
        path,
        sep="\t",
        names=RATING_COLUMNS,
        dtype={"user_id": str, "movie_id": str, "rating": float, "timestamp": int},
    )


def load_triplets(download: bool = True) -> List[Tuple[str, str, float]]:
    """Load ratings as a list of ``(user_id, movie_id, rating)`` triplets."""
    frame = load_ratings(download=download)
    return list(zip(frame["user_id"], frame["movie_id"], frame["rating"]))
