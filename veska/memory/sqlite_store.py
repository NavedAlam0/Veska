"""
SQLite-based memory store.

Full history, SQL searchable. No compression needed — SQLite handles
large datasets efficiently. Uses Python's built-in sqlite3.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from veska.memory.store import Memory, MemoryStore


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed memory store. Stores everything, searches with SQL."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        """Create table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                timestamp REAL NOT NULL,
                PRIMARY KEY (key, timestamp)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
        """)
        self._conn.commit()

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            key=row["key"],
            value=row["value"],
            metadata=json.loads(row["metadata"]),
            timestamp=row["timestamp"],
        )

    async def save(self, key: str, value: str, metadata: Optional[dict] = None) -> None:
        self._conn.execute(
            "INSERT INTO memories (key, value, metadata, timestamp) VALUES (?, ?, ?, ?)",
            (key, value, json.dumps(metadata or {}), time.time()),
        )
        self._conn.commit()

    async def load(self, key: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE key = ? ORDER BY timestamp DESC LIMIT 1",
            (key,),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    async def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Search by matching query words against key and value."""
        words = query.lower().split()
        if not words:
            return []

        # Build WHERE clause: each word must appear in key OR value
        conditions = []
        params = []
        for word in words:
            conditions.append("(LOWER(key) LIKE ? OR LOWER(value) LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%"])

        # Score by number of matching words (use OR, sort by match count)
        where = " OR ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self._conn.commit()

    async def list_all(self) -> list[Memory]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC"
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
