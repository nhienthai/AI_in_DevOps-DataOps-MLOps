"""
Tests for the MovieLens 100K downloader.

The download is the pipeline's only external dependency and the one that has
actually broken CI: upstream let its TLS certificate expire and every training
run died on ``CERTIFICATE_VERIFY_FAILED``. These tests pin the behaviour that
keeps that from being fatal - and the checksum guard that is the only reason
the certificate fallback is acceptable.

Every test here is hermetic: no test touches the network.

Run tests:
    pytest tests/data/test_dataset_download.py -v
"""

import hashlib
import io
import ssl
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest

from training import dataset

RATINGS_LINE = "196\t242\t3\t881250949\n"


def make_archive(u_data: str = RATINGS_LINE, extra_member: str = "") -> bytes:
    """Build an in-memory ml-100k-shaped zip."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(dataset.U_DATA_MEMBER, u_data)
        archive.writestr("ml-100k/u.item", "1|Toy Story (1995)\n")
        if extra_member:
            archive.writestr(extra_member, "payload")
    return buffer.getvalue()


@pytest.fixture
def pinned_archive(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """A fake archive whose ratings file matches the pinned checksum."""
    payload = make_archive()
    digest = hashlib.sha256(RATINGS_LINE.encode()).hexdigest()
    monkeypatch.setattr(dataset, "U_DATA_SHA256", digest)
    return payload


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the dataset cache at a throwaway directory."""
    monkeypatch.setenv("SURPRISE_DATA_FOLDER", str(tmp_path))
    monkeypatch.delenv("ML100K_ZIP", raising=False)
    monkeypatch.delenv("ML100K_URL", raising=False)
    return tmp_path


class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


def expired_certificate_error() -> urllib.error.URLError:
    """Reproduce the exact error the expired upstream certificate raises."""
    cert_error = ssl.SSLCertVerificationError("certificate verify failed: certificate has expired")
    cert_error.verify_message = "certificate has expired"
    return urllib.error.URLError(cert_error)


# =============================================================================
# Integrity guards - these are what make the fallback safe
# =============================================================================


class TestArchiveVerification:
    """The archive is only trusted once its ratings file matches the pin."""

    def test_accepts_the_official_checksum(self, pinned_archive: bytes) -> None:
        """An archive matching the pinned digest opens normally."""
        with dataset._verified_archive(pinned_archive) as archive:
            assert dataset.U_DATA_MEMBER in archive.namelist()

    def test_rejects_tampered_ratings(self) -> None:
        """A ratings file that is not the official one is refused."""
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            dataset._verified_archive(make_archive("1\t1\t5\t0\n"))

    def test_rejects_path_traversal(self, pinned_archive: bytes) -> None:
        """A member escaping the target directory is refused before extraction."""
        payload = make_archive(extra_member="../evil.sh")
        with pytest.raises(RuntimeError, match="unsafe archive member"):
            dataset._verified_archive(payload)

    def test_rejects_absolute_member(self, pinned_archive: bytes) -> None:
        """An absolute member path is refused too."""
        payload = make_archive(extra_member="/etc/evil.sh")
        with pytest.raises(RuntimeError, match="unsafe archive member"):
            dataset._verified_archive(payload)


# =============================================================================
# The expired-certificate fallback
# =============================================================================


class TestExpiredCertificateFallback:
    """An expired *server* certificate must not fail the training run."""

    def test_retries_without_verification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second attempt passes an unverified SSL context."""
        contexts = []

        def fake_urlopen(url: str, timeout: int = 0, context: Any = None) -> FakeResponse:
            contexts.append(context)
            if context is None:
                raise expired_certificate_error()
            return FakeResponse(b"payload")

        monkeypatch.setattr(dataset.urllib.request, "urlopen", fake_urlopen)

        assert dataset._download(dataset.ML100K_URL) == b"payload"
        assert contexts[0] is None, "the first attempt must verify the certificate"
        assert contexts[1].verify_mode == ssl.CERT_NONE

    def test_other_network_errors_still_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only certificate expiry is tolerated - a 404 or DNS failure is not."""

        def fake_urlopen(url: str, timeout: int = 0, context: Any = None) -> FakeResponse:
            raise urllib.error.URLError("Name or service not known")

        monkeypatch.setattr(dataset.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(urllib.error.URLError):
            dataset._download(dataset.ML100K_URL)

    def test_end_to_end_over_expired_certificate(
        self, cache_dir: Path, pinned_archive: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_ml100k caches the dataset despite the expired certificate."""

        def fake_urlopen(url: str, timeout: int = 0, context: Any = None) -> FakeResponse:
            if context is None:
                raise expired_certificate_error()
            return FakeResponse(pinned_archive)

        monkeypatch.setattr(dataset.urllib.request, "urlopen", fake_urlopen)

        path = dataset.ensure_ml100k()
        assert path == cache_dir / "ml-100k" / "ml-100k" / "u.data"
        assert path.read_text() == RATINGS_LINE


# =============================================================================
# Cache and offline escape hatches
# =============================================================================


class TestCacheAndOverrides:
    """The download is skipped when it can be."""

    def test_cached_dataset_is_not_redownloaded(
        self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached u.data short-circuits the network entirely."""
        path = cache_dir / "ml-100k" / "ml-100k" / "u.data"
        path.parent.mkdir(parents=True)
        path.write_text(RATINGS_LINE)

        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("network access on a cache hit")

        monkeypatch.setattr(dataset.urllib.request, "urlopen", explode)

        assert dataset.ensure_ml100k() == path

    def test_local_zip_env_var_is_used(
        self,
        cache_dir: Path,
        pinned_archive: bytes,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ML100K_ZIP lets an air-gapped run supply the archive itself."""
        local_zip = tmp_path / "ml-100k.zip"
        local_zip.write_bytes(pinned_archive)
        monkeypatch.setenv("ML100K_ZIP", str(local_zip))

        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("network access despite ML100K_ZIP")

        monkeypatch.setattr(dataset.urllib.request, "urlopen", explode)

        assert dataset.ensure_ml100k().read_text() == RATINGS_LINE

    def test_url_override_is_tried_first(
        self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ML100K_URL takes priority but keeps upstream as a fallback."""
        monkeypatch.setenv("ML100K_URL", "https://mirror.example/ml-100k.zip")
        assert dataset.download_urls() == [
            "https://mirror.example/ml-100k.zip",
            dataset.ML100K_URL,
        ]

    def test_failure_message_names_every_url(
        self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When nothing works the error lists what was tried and how to recover."""
        monkeypatch.setenv("ML100K_URL", "https://mirror.example/ml-100k.zip")

        def fake_urlopen(url: str, timeout: int = 0, context: Any = None) -> FakeResponse:
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(dataset.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(RuntimeError) as exc_info:
            dataset.ensure_ml100k()

        message = str(exc_info.value)
        assert "mirror.example" in message
        assert dataset.ML100K_URL in message
        assert "ML100K_ZIP" in message
