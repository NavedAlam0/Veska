"""
File-based session store.

Each session is a JSON file: {directory}/{user_id}/{session_id}.json
Simple, zero setup. Good for development and small projects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veska.providers.base import Message
from veska.sessions.store import SessionStore


class FileSessionStore(SessionStore):
    """JSON file-based session store."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, user_id: str, session_id: str) -> Path:
        user_dir = self._dir / _safe_name(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{_safe_name(session_id)}.json"

    async def load(self, user_id: str, session_id: str) -> list[Message]:
        path = self._session_path(user_id, session_id)
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [Message(**msg) for msg in data]
        except (json.JSONDecodeError, OSError):
            return []

    async def save(self, user_id: str, session_id: str, messages: list[Message]) -> None:
        path = self._session_path(user_id, session_id)
        data = [_message_to_dict(msg) for msg in messages]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def delete(self, user_id: str, session_id: str) -> None:
        path = self._session_path(user_id, session_id)
        path.unlink(missing_ok=True)

    async def list_sessions(self, user_id: str) -> list[str]:
        user_dir = self._dir / _safe_name(user_id)
        if not user_dir.exists():
            return []
        return [p.stem for p in user_dir.glob("*.json")]


def _safe_name(name: str) -> str:
    """Convert a name to a safe directory/filename."""
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("..", "_")


def _message_to_dict(msg: Message) -> dict:
    """Serialize a Message, handling both str and list content."""
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    return d
