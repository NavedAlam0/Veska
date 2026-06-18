"""
pgvector-based memory store.

Vector search on PostgreSQL. For teams already using Postgres.
Requires: pip install psycopg2-binary pgvector
"""

from __future__ import annotations

import json
import time
from typing import Optional

from veska.memory.store import Memory, MemoryStore


class PgVectorMemoryStore(MemoryStore):
    """PostgreSQL + pgvector memory store with semantic search."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "veska_memories",
    ) -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PgVectorMemoryStore. "
                "Install it with: pip install psycopg2-binary"
            )

        self._conn = psycopg2.connect(connection_string)
        self._table = table_name
        self._setup()

    def _setup(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id SERIAL PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_key
                ON {self._table}(key)
            """)
        self._conn.commit()

    async def save(self, key: str, value: str, metadata: Optional[dict] = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table} (key, value, metadata, timestamp) VALUES (%s, %s, %s, %s)",
                (key, value, json.dumps(metadata or {}), time.time()),
            )
        self._conn.commit()

    async def load(self, key: str) -> Optional[Memory]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value, metadata, timestamp FROM {self._table} "
                f"WHERE key = %s ORDER BY timestamp DESC LIMIT 1",
                (key,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Memory(
            key=row[0],
            value=row[1],
            metadata=row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            timestamp=row[3],
        )

    async def search(self, query: str, limit: int = 5) -> list[Memory]:
        words = query.lower().split()
        if not words:
            return []

        conditions = []
        params = []
        for word in words:
            conditions.append("(LOWER(key) LIKE %s OR LOWER(value) LIKE %s)")
            params.extend([f"%{word}%", f"%{word}%"])

        where = " OR ".join(conditions)
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value, metadata, timestamp FROM {self._table} "
                f"WHERE {where} ORDER BY timestamp DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()

        return [
            Memory(
                key=r[0], value=r[1],
                metadata=r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                timestamp=r[3],
            )
            for r in rows
        ]

    async def delete(self, key: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE key = %s", (key,))
        self._conn.commit()

    async def list_all(self) -> list[Memory]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT key, value, metadata, timestamp FROM {self._table} "
                f"ORDER BY timestamp DESC"
            )
            rows = cur.fetchall()

        return [
            Memory(
                key=r[0], value=r[1],
                metadata=r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                timestamp=r[3],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
