"""
File-based cache.

Persists cache entries as JSON files on disk. Survives program restarts.
Each entry is a separate file for fast read/write without loading everything.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from veska.cache.store import CacheStore

DEFAULT_TTL = 3600


class FileCache(CacheStore):
    """File-based cache with TTL expiry. Each key is a separate JSON file."""

    def __init__(self, directory: str, default_ttl: int = DEFAULT_TTL) -> None:
        self.default_ttl = default_ttl
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        """Convert key to a safe filename."""
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        # Use hash for long keys
        if len(safe) > 100:
            import hashlib
            safe = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{safe}.json"

    def _read_entry(self, key: str) -> Optional[dict]:
        """Read and validate a cache entry from disk."""
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        expires_at = data.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            path.unlink(missing_ok=True)
            return None

        return data

    async def get(self, key: str) -> Any:
        entry = self._read_entry(key)
        return entry["value"] if entry else None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expires_at = None if ttl == 0 else time.time() + ttl

        data = {
            "key": key,
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time(),
        }

        path = self._key_path(key)
        path.write_text(json.dumps(data), encoding="utf-8")

    async def delete(self, key: str) -> None:
        path = self._key_path(key)
        path.unlink(missing_ok=True)

    async def clear(self) -> None:
        for path in self._dir.glob("*.json"):
            path.unlink(missing_ok=True)

    async def exists(self, key: str) -> bool:
        return self._read_entry(key) is not None
