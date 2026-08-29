"""
MovieLens 100K dataset access.

Downloads and caches the dataset in the *same* directory scikit-surprise uses
(``~/.surprise_data/ml-100k/ml-100k/u.data``), so both training backends and the
data-quality tests share one cache and one copy on disk.

The download is deliberately defensive: ``files.grouplens.org`` is a single
upstream with no mirror, and its TLS certificate has lapsed before (it expired
on 2026-08-28 and broke every CI run). Every payload is therefore checked
against a pinned SHA-256 of ``u.data`` before it is installed into the cache,
which is what makes the expired-certificate fallback below safe.
"""

import hashlib
import io
import os
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

#: SHA-256 of ``ml-100k/u.data`` inside the official archive. Pinning the
#: ratings file (rather than the zip) keeps the check valid for any mirror.
U_DATA_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"

#: Member of the archive holding the ratings.
U_DATA_MEMBER = "ml-100k/u.data"

DOWNLOAD_TIMEOUT = 120

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


def download_urls() -> List[str]:
    """Return the URLs to try, in order. The ML100K_URL env var takes priority."""
    override = os.getenv("ML100K_URL")
    return [override, ML100K_URL] if override else [ML100K_URL]


def _certificate_error(exc: BaseException) -> Optional[ssl.SSLCertVerificationError]:
    """Return the certificate error behind ``exc``, if that is what it is."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return exc
    reason = getattr(exc, "reason", None)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return reason
    return None


def _download(url: str) -> bytes:
    """
    Fetch ``url``, tolerating an expired *server* certificate.

    Retrying without verification is only acceptable because
    :func:`ensure_ml100k` refuses to install any payload whose ratings file does
    not match :data:`U_DATA_SHA256`, so a tampered response is rejected no
    matter what the transport proved. As soon as upstream renews its
    certificate the verified path succeeds again with no code change.
    """
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            # urlopen is typed as returning Any, so pin the payload type here.
            payload: bytes = response.read()
        return payload
    except (urllib.error.URLError, ssl.SSLError) as exc:
        cert_error = _certificate_error(exc)
        if cert_error is None:
            raise
        print(
            f"      TLS verification failed for {url} ({cert_error.verify_message}); "
            "retrying with checksum verification only."
        )
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT, context=context) as response:
            fallback_payload: bytes = response.read()
        return fallback_payload


def _fetch_archive() -> bytes:
    """Return the ml-100k zip bytes from a local file or the first URL that works."""
    local_zip = os.getenv("ML100K_ZIP")
    if local_zip:
        return Path(local_zip).read_bytes()

    failures = []
    for url in download_urls():
        try:
            return _download(url)
        except Exception as exc:  # noqa: BLE001 - every failure is reported below
            failures.append(f"  {url}: {exc}")

    raise RuntimeError(
        "Could not download MovieLens 100K:\n"
        + "\n".join(failures)
        + "\n\nWork around it by pointing ML100K_ZIP at a local copy of ml-100k.zip, "
        "or ML100K_URL at a mirror."
    )


def _verified_archive(payload: bytes) -> zipfile.ZipFile:
    """Open the archive, rejecting unsafe members and an unexpected ratings file."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    try:
        for name in archive.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Refusing to extract unsafe archive member: {name}")

        digest = hashlib.sha256(archive.read(U_DATA_MEMBER)).hexdigest()
        if digest != U_DATA_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for {U_DATA_MEMBER}: expected {U_DATA_SHA256}, "
                f"got {digest}. The download was corrupt or is not the official "
                "MovieLens 100K archive."
            )
    except Exception:
        archive.close()
        raise
    return archive


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

    with _verified_archive(_fetch_archive()) as archive:
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
