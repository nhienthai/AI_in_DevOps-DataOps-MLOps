"""
Optional Redis cache for predictions.

The cache is a pure optimisation: every operation fails open. If Redis is down,
misconfigured, or simply not running, the API keeps serving predictions from the
model and only loses the speedup. That is why nothing here raises.
"""

import logging
import time
from typing import Optional

from app.config import (
    CACHE_ENABLED,
    CACHE_TIMEOUT_SECONDS,
    CACHE_TTL_SECONDS,
    MODEL_VERSION,
    REDIS_URL,
)

logger = logging.getLogger(__name__)

# How long to stop touching Redis after a failure, so a dead Redis does not add
# a timeout to every single request.
_BACKOFF_SECONDS = 30.0


class PredictionCache:
    """Read-through cache for (user_id, movie_id) -> predicted rating."""

    def __init__(
        self,
        enabled: bool = CACHE_ENABLED,
        url: str = REDIS_URL,
        ttl: int = CACHE_TTL_SECONDS,
        timeout: float = CACHE_TIMEOUT_SECONDS,
    ):
        self.enabled = enabled
        self.url = url
        self.ttl = ttl
        self.timeout = timeout
        self._client = None
        self._degraded_until = 0.0

        if not self.enabled:
            logger.info("Prediction cache disabled (CACHE_ENABLED=false)")
            return

        try:
            import redis  # imported lazily: only needed when caching is on

            self._client = redis.Redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
            )
            self._client.ping()
            logger.info(f"Prediction cache connected to {self.url} (ttl={self.ttl}s)")
        except Exception as e:
            # Not fatal: mark degraded and let the periodic retry pick it up.
            logger.warning(f"Cache unavailable at startup, serving without it: {e}")
            self._degrade()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _degrade(self) -> None:
        """Skip Redis for a while after a failure."""
        self._degraded_until = time.monotonic() + _BACKOFF_SECONDS

    def _usable(self) -> bool:
        return (
            self.enabled
            and self._client is not None
            and time.monotonic() >= self._degraded_until
        )

    @staticmethod
    def _key(user_id: str, movie_id: str) -> str:
        # The model version is part of the key so retraining invalidates old
        # entries instead of serving stale ratings.
        return f"pred:{MODEL_VERSION}:{user_id}:{movie_id}"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def get(self, user_id: str, movie_id: str) -> Optional[float]:
        """Return the cached rating, or None on miss / any cache problem."""
        if not self._usable():
            return None
        try:
            value = self._client.get(self._key(user_id, movie_id))
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None  # corrupted entry: treat as a miss
        except Exception as e:
            logger.warning(f"Cache read failed, falling back to the model: {e}")
            self._degrade()
            return None

    def set(self, user_id: str, movie_id: str, rating: float) -> None:
        """Store a rating. Failures are logged and ignored."""
        if not self._usable():
            return
        try:
            self._client.setex(self._key(user_id, movie_id), self.ttl, rating)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
            self._degrade()

    def is_connected(self) -> bool:
        """Whether the cache is currently serving (used by /health)."""
        if not self._usable():
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            self._degrade()
            return False


# =============================================================================
# Singleton
# =============================================================================
_cache_instance: Optional[PredictionCache] = None


def get_cache() -> PredictionCache:
    """Get or create the cache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PredictionCache()
    return _cache_instance
