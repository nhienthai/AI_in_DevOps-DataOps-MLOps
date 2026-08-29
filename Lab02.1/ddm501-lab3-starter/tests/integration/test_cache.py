"""
Integration tests for the optional Redis prediction cache.

The cache's defining property is that it **fails open**: if Redis is missing,
slow, or broken, the API must keep serving predictions and only lose the
speedup. That property is what these tests are for - a cache that turns a
dependency outage into an API outage is worse than no cache at all.

Tests that need a live Redis skip when one is not reachable; CI provides one as
a service container, so they run there.

Run tests:
    pytest tests/integration/test_cache.py -v
"""

import os
from typing import Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from app.cache import PredictionCache, get_cache, reset_cache
from app.config import MODEL_VERSION

pytestmark = pytest.mark.integration

# An address nothing listens on, used to prove the failure paths.
UNREACHABLE_URL = "redis://127.0.0.1:6399/0"

REDIS_URL = os.getenv("TEST_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))


def _redis_available() -> bool:
    """Whether a real Redis is reachable for the live-cache tests."""
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
        return bool(client.ping())
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(), reason=f"No Redis reachable at {REDIS_URL}"
)


@pytest.fixture
def live_cache() -> Iterator[PredictionCache]:
    """A cache backed by a real Redis, flushed around the test."""
    cache = PredictionCache(enabled=True, url=REDIS_URL, ttl=60)
    yield cache
    try:
        import redis

        redis.Redis.from_url(REDIS_URL).flushdb()
    except Exception:  # pragma: no cover - cleanup is best effort
        pass


class TestCacheDisabled:
    """Default configuration: the cache is off."""

    def test_cache_disabled_by_default(self) -> None:
        """CACHE_ENABLED defaults to false, so nothing touches Redis."""
        cache = PredictionCache(enabled=False)
        assert cache.enabled is False
        assert cache.is_connected() is False

    def test_disabled_cache_always_misses(self) -> None:
        """A disabled cache is a no-op, not an error."""
        cache = PredictionCache(enabled=False)
        cache.set("196", "242", 3.9)
        assert cache.get("196", "242") is None

    def test_disabled_cache_never_imports_redis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        The redis client is imported lazily.

        Breaking the import proves a disabled cache never reaches for it - which
        is what keeps redis an optional dependency.
        """
        import builtins

        real_import = builtins.__import__

        def exploding_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "redis":
                raise ImportError("redis must not be imported when the cache is off")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", exploding_import)
        assert PredictionCache(enabled=False).enabled is False


class TestCacheFailsOpen:
    """The cache must never turn a Redis problem into an API problem."""

    def test_unreachable_redis_does_not_raise(self) -> None:
        """Construction against a dead address degrades instead of exploding."""
        cache = PredictionCache(enabled=True, url=UNREACHABLE_URL, timeout=0.05)
        assert cache.is_connected() is False

    def test_get_on_unreachable_redis_returns_none(self) -> None:
        """A read failure looks exactly like a miss to the caller."""
        cache = PredictionCache(enabled=True, url=UNREACHABLE_URL, timeout=0.05)
        assert cache.get("196", "242") is None

    def test_set_on_unreachable_redis_is_silent(self) -> None:
        """A write failure is logged and swallowed - the prediction still returns."""
        cache = PredictionCache(enabled=True, url=UNREACHABLE_URL, timeout=0.05)
        cache.set("196", "242", 3.9)

    def test_failure_triggers_backoff(self) -> None:
        """
        After a failure the cache stops touching Redis for a while.

        Without backoff, a dead Redis would add its connect timeout to every
        single request - turning a lost optimisation into a latency incident.
        """
        cache = PredictionCache(enabled=True, url=UNREACHABLE_URL, timeout=0.05)
        assert cache._usable() is False

    def test_api_serves_predictions_without_redis(
        self, api_client: TestClient, sample_prediction_request: Dict[str, str]
    ) -> None:
        """End to end: no Redis in this environment, predictions still 200."""
        response = api_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_health_is_healthy_without_cache(self, test_client: TestClient) -> None:
        """
        A missing cache must NOT make the service unhealthy.

        Only the model gates health; the cache is an optimisation.
        """
        data = test_client.get("/health").json()
        assert data["status"] == ("healthy" if data["model_loaded"] else "unhealthy")
        assert isinstance(data["cache_connected"], bool)


class _StubRedis:
    """
    In-memory stand-in for a Redis client.

    Lets the read/write paths be exercised deterministically without a server,
    including the failure branches a live Redis will not reproduce on demand.
    """

    def __init__(self, fail_on: str = "", stored: Dict[str, str] | None = None) -> None:
        self.fail_on = fail_on
        self.stored: Dict[str, str] = stored or {}
        self.setex_calls: list = []

    def _maybe_fail(self, operation: str) -> None:
        if self.fail_on == operation:
            raise ConnectionError(f"stub redis failing on {operation}")

    def get(self, key: str) -> str | None:
        self._maybe_fail("get")
        return self.stored.get(key)

    def setex(self, key: str, ttl: int, value: float) -> None:
        self._maybe_fail("setex")
        self.setex_calls.append((key, ttl, value))
        self.stored[key] = str(value)

    def ping(self) -> bool:
        self._maybe_fail("ping")
        return True


def _cache_with(client: _StubRedis) -> PredictionCache:
    """An enabled cache wired to a stub client, past any startup backoff."""
    cache = PredictionCache(enabled=False)
    cache.enabled = True
    cache._client = client
    cache._degraded_until = 0.0
    return cache


class TestCacheReadWritePaths:
    """Read/write behaviour, driven with a stub client so every branch runs."""

    def test_hit_returns_the_stored_float(self) -> None:
        """A stored value comes back parsed as a float."""
        key = PredictionCache._key("196", "242")
        cache = _cache_with(_StubRedis(stored={key: "3.97"}))
        assert cache.get("196", "242") == 3.97

    def test_miss_returns_none(self) -> None:
        """An absent key is a miss."""
        cache = _cache_with(_StubRedis())
        assert cache.get("196", "242") is None

    def test_corrupted_value_is_treated_as_a_miss(self) -> None:
        """A non-numeric value must not raise into the request handler."""
        key = PredictionCache._key("196", "242")
        cache = _cache_with(_StubRedis(stored={key: "not-a-number"}))
        assert cache.get("196", "242") is None

    def test_read_failure_returns_none_and_degrades(self) -> None:
        """A read error looks like a miss, and stops further Redis calls."""
        cache = _cache_with(_StubRedis(fail_on="get"))
        assert cache.get("196", "242") is None
        assert cache._usable() is False

    def test_set_writes_with_the_configured_ttl(self) -> None:
        """Entries expire; a cache without a TTL is a memory leak."""
        client = _StubRedis()
        cache = _cache_with(client)
        cache.ttl = 123
        cache.set("196", "242", 3.97)
        assert client.setex_calls == [(PredictionCache._key("196", "242"), 123, 3.97)]

    def test_write_failure_is_swallowed_and_degrades(self) -> None:
        """A failed write never propagates - the prediction still returns."""
        cache = _cache_with(_StubRedis(fail_on="setex"))
        cache.set("196", "242", 3.97)
        assert cache._usable() is False

    def test_is_connected_true_when_ping_succeeds(self) -> None:
        """A responsive client reports connected."""
        assert _cache_with(_StubRedis()).is_connected() is True

    def test_is_connected_false_when_ping_fails(self) -> None:
        """A failing ping reports disconnected and degrades."""
        cache = _cache_with(_StubRedis(fail_on="ping"))
        assert cache.is_connected() is False
        assert cache._usable() is False

    def test_degraded_cache_skips_redis_entirely(self) -> None:
        """
        While degraded, no call reaches the client at all.

        That is the whole point of the backoff: a dead Redis must not add its
        timeout to every request.
        """
        client = _StubRedis()
        cache = _cache_with(client)
        cache._degrade()
        assert cache.get("196", "242") is None
        cache.set("196", "242", 3.97)
        assert client.setex_calls == []


class TestApiUsesTheCache:
    """The API must actually consult the cache, not merely own one."""

    def test_cache_hit_skips_the_model(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """
        On a hit the model is never called.

        Without this the cache could be silently dead - every request would
        still return the right answer, just slowly, and no test would notice.
        """

        class CountingModel:
            calls = 0

            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                CountingModel.calls += 1
                return 1.0

        class HitCache:
            enabled = True

            def get(self, user_id: str, movie_id: str) -> float:
                return 4.25

            def set(self, user_id: str, movie_id: str, rating: float) -> None:
                raise AssertionError("a hit must not write back")

            def is_connected(self) -> bool:
                return True

        monkeypatch.setattr("app.main.model", CountingModel())
        monkeypatch.setattr("app.main.cache", HitCache())

        response = test_client.post("/predict", json=sample_prediction_request)

        assert response.status_code == 200
        assert response.json()["predicted_rating"] == 4.25
        assert CountingModel.calls == 0

    def test_cache_miss_writes_back(
        self,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        sample_prediction_request: Dict[str, str],
    ) -> None:
        """A miss computes the rating and stores it, so the next call hits."""
        writes: list = []

        # The sibling hit-test stubs the model; this one did not, so it was the
        # only test in the file that needed a trained svd_model.pkl on disk.
        # Without it /predict answers 503 and the test fails rather than skips,
        # which makes a fresh clone look broken instead of untrained.
        class StubModel:
            def is_loaded(self) -> bool:
                return True

            def predict(self, user_id: str, movie_id: str) -> float:
                return 3.75

        class MissCache:
            enabled = True

            def get(self, user_id: str, movie_id: str) -> None:
                return None

            def set(self, user_id: str, movie_id: str, rating: float) -> None:
                writes.append((user_id, movie_id, rating))

            def is_connected(self) -> bool:
                return True

        monkeypatch.setattr("app.main.model", StubModel())
        monkeypatch.setattr("app.main.cache", MissCache())

        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert len(writes) == 1
        # Written value is the model's output, not merely whatever was returned.
        assert writes[0][2] == 3.75
        assert writes[0][2] == response.json()["predicted_rating"]

    def test_health_reports_cache_connected(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reachable cache shows up in /health."""

        class ConnectedCache:
            enabled = True

            def is_connected(self) -> bool:
                return True

        monkeypatch.setattr("app.main.cache", ConnectedCache())
        assert test_client.get("/health").json()["cache_connected"] is True


class TestCacheKeying:
    """Key construction - what invalidates and what collides."""

    def test_key_includes_model_version(self) -> None:
        """
        Retraining must not serve stale ratings.

        Versioning the key means a new model starts with a cold cache instead of
        inheriting the previous model's answers.
        """
        key = PredictionCache._key("196", "242")
        assert MODEL_VERSION in key
        assert key.endswith(":196:242")

    def test_different_pairs_get_different_keys(self) -> None:
        """No collisions between distinct user-movie pairs."""
        assert PredictionCache._key("196", "242") != PredictionCache._key("242", "196")


class TestCacheSingleton:
    """get_cache()/reset_cache() lifecycle."""

    def test_get_cache_returns_same_instance(self) -> None:
        """One shared connection pool, not one per request."""
        reset_cache()
        assert get_cache() is get_cache()
        reset_cache()

    def test_reset_cache_clears_instance(self) -> None:
        """reset_cache() forces a rebuild, so tests cannot leak state."""
        reset_cache()
        first = get_cache()
        reset_cache()
        assert get_cache() is not first
        reset_cache()


@requires_redis
class TestCacheWithLiveRedis:
    """Round-trip behaviour against a real Redis (skipped when none is running)."""

    def test_connects(self, live_cache: PredictionCache) -> None:
        """A reachable Redis reports connected."""
        assert live_cache.is_connected() is True

    def test_set_then_get_round_trips(self, live_cache: PredictionCache) -> None:
        """A stored rating comes back as the same float."""
        live_cache.set("196", "242", 3.97)
        assert live_cache.get("196", "242") == 3.97

    def test_miss_returns_none(self, live_cache: PredictionCache) -> None:
        """An unseen pair is a miss, not a zero."""
        assert live_cache.get("no_such_user", "no_such_movie") is None

    def test_corrupted_entry_is_treated_as_a_miss(self, live_cache: PredictionCache) -> None:
        """
        A non-numeric value in Redis must not crash the request.

        Anything can write to a shared cache; the reader stays defensive.
        """
        import redis

        redis.Redis.from_url(REDIS_URL, decode_responses=True).set(
            PredictionCache._key("196", "242"), "not-a-number"
        )
        assert live_cache.get("196", "242") is None

    def test_cached_prediction_matches_the_model(
        self, live_cache: PredictionCache, trained_model: object
    ) -> None:
        """The cache returns what the model produced, not a rounded copy."""
        rating = trained_model.predict("196", "242")  # type: ignore[attr-defined]
        live_cache.set("196", "242", rating)
        assert live_cache.get("196", "242") == rating


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
