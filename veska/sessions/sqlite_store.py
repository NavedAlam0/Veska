"""
SQLite-based session store.

All sessions in one database file. Handles many users/sessions efficiently.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from veska.providers.base import Message
from veska.sessions.store import SessionStore


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._setup()

    def _setup(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, session_id)
            )
        """)
        self._conn.commit()

    async def load(self, user_id: str, session_id: str) -> list[Message]:
        row = self._conn.execute(
            "SELECT messages FROM sessions WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()

        if not row:
            return []

        try:
            data = json.loads(row[0])
            return [Message(**msg) for msg in data]
        except (json.JSONDecodeError, ValueError):
            return []

    async def save(self, user_id: str, session_id: str, messages: list[Message]) -> None:
        import time
        data = [_message_to_dict(msg) for msg in messages]

        self._conn.execute(
            """INSERT INTO sessions (user_id, session_id, messages, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, session_id)
               DO UPDATE SET messages = excluded.messages, updated_at = excluded.updated_at""",
            (user_id, session_id, json.dumps(data), time.time()),
        )
        self._conn.commit()

    async def delete(self, user_id: str, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        self._conn.commit()

    async def list_sessions(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()


def _message_to_dict(msg: Message) -> dict:
    """Serialize a Message, handling both str and list content."""
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    return d
