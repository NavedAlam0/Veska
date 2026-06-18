"""
In-memory cache.

Fast, simple, gone when program closes.
Default TTL is 3600s (1 hour), developer can override globally or per-entry.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from veska.cache.store import CacheStore

DEFAULT_TTL = 3600  # 1 hour


class InMemoryCache(CacheStore):
    """In-memory cache with TTL expiry."""

    def __init__(self, default_ttl: int = DEFAULT_TTL) -> None:
        self.default_ttl = default_ttl
        self._store: dict[str, tuple[Any, Optional[float]]] = {}
        # _store maps key -> (value, expires_at)
        # expires_at is None for permanent entries

    def _is_expired(self, key: str) -> bool:
        if key not in self._store:
            return True
        _, expires_at = self._store[key]
        if expires_at is None:
            return False
        return time.time() > expires_at

    def _cleanup_key(self, key: str) -> None:
        if key in self._store and self._is_expired(key):
            del self._store[key]

    async def get(self, key: str) -> Any:
        self._cleanup_key(key)
        if key not in self._store:
            return None
        return self._store[key][0]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expires_at = None if ttl == 0 else time.time() + ttl
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()

    async def exists(self, key: str) -> bool:
        self._cleanup_key(key)
        return key in self._store
