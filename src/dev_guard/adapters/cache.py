from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError


class RedisCache(Cache):
    """Lightweight Redis adapter; requires the redis extra to be installed."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover - optional import
            raise RuntimeError(
                "redis extra not installed. Install with: pip install dev-guard[cache]"
            ) from exc
        self._redis = redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Optional[Any]:
        return self._redis.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is not None:
            self._redis.setex(key, ttl_seconds, value)
        else:
            self._redis.set(key, value)

    def delete(self, key: str) -> None:
        self._redis.delete(key)

