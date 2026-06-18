"""
File-based memory store.

Persists memories as JSON files. Uses 3-level compression
so the file stays small even after thousands of entries.

Level 1 (Recent):  Full detailed entries      — last 50
Level 2 (Older):   Compressed entries         — entries 50-200 (key + truncated value)
Level 3 (Oldest):  Metadata only              — everything older (key + category + timestamp)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from veska.memory.store import Memory, MemoryStore

# Compression thresholds
LEVEL_1_LIMIT = 50   # Full detail
LEVEL_2_LIMIT = 200  # Compressed
LEVEL_2_VALUE_TRUNCATE = 100  # Truncate value to this many chars in level 2


class FileMemoryStore(MemoryStore):
    """JSON file-based memory store with 3-level compression."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "memories.json"
        self._memories: list[Memory] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load from disk on first access."""
        if self._loaded:
            return
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._memories = [Memory(**entry) for entry in data]
            except (json.JSONDecodeError, OSError):
                self._memories = []
        self._loaded = True

    def _persist(self) -> None:
        """Write memories to disk with 3-level compression."""
        self._memories = self._compress(self._memories)
        data = [m.model_dump() for m in self._memories]
        self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _compress(self, memories: list[Memory]) -> list[Memory]:
        """Apply 3-level compression to keep file small."""
        if len(memories) <= LEVEL_1_LIMIT:
            return memories  # All recent, no compression needed

        # Sort by timestamp (newest first)
        sorted_mems = sorted(memories, key=lambda m: m.timestamp, reverse=True)

        result = []

        for i, mem in enumerate(sorted_mems):
            if i < LEVEL_1_LIMIT:
                # Level 1: full detail
                result.append(mem)
            elif i < LEVEL_2_LIMIT:
                # Level 2: truncated value
                result.append(Memory(
                    key=mem.key,
                    value=mem.value[:LEVEL_2_VALUE_TRUNCATE],
                    metadata={**mem.metadata, "_compressed": "level2"},
                    timestamp=mem.timestamp,
                ))
            else:
                # Level 3: metadata only
                result.append(Memory(
                    key=mem.key,
                    value="",
                    metadata={**mem.metadata, "_compressed": "level3"},
                    timestamp=mem.timestamp,
                ))

        return result

    async def save(self, key: str, value: str, metadata: Optional[dict] = None) -> None:
        self._ensure_loaded()
        entry = Memory(
            key=key,
            value=value,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self._memories.append(entry)
        self._persist()

    async def load(self, key: str) -> Optional[Memory]:
        self._ensure_loaded()
        for mem in reversed(self._memories):  # Most recent first
            if mem.key == key:
                return mem
        return None

    async def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Keyword search — matches query words against key and value."""
        self._ensure_loaded()
        query_lower = query.lower()
        query_words = query_lower.split()

        scored: list[tuple[int, Memory]] = []
        for mem in self._memories:
            text = f"{mem.key} {mem.value}".lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [mem for _, mem in scored[:limit]]

    async def delete(self, key: str) -> None:
        self._ensure_loaded()
        self._memories = [m for m in self._memories if m.key != key]
        self._persist()

    async def list_all(self) -> list[Memory]:
        self._ensure_loaded()
        return list(self._memories)
