"""
CacheStore interface.

General-purpose key-value store with TTL.
Not locked to any type — cache anything.
Community can build Redis, Memcached, etc. — just implement these 5 methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheStore(ABC):
    """Abstract base for all cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Any:
        """Get a cached value. Returns None if not found or expired."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value. ttl=None uses the store's default TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a cached entry."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        ...
