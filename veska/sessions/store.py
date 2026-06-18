"""
SessionStore interface.

Persists conversation threads using user_id + session_id.
Same session_id resumes the chat, new one starts fresh.
Community can build Redis, PostgreSQL, etc. — just implement these methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from veska.providers.base import Message


class SessionStore(ABC):
    """Abstract base for all session backends."""

    @abstractmethod
    async def load(self, user_id: str, session_id: str) -> list[Message]:
        """Load conversation history for a session. Returns empty list if not found."""
        ...

    @abstractmethod
    async def save(self, user_id: str, session_id: str, messages: list[Message]) -> None:
        """Save conversation history for a session."""
        ...

    @abstractmethod
    async def delete(self, user_id: str, session_id: str) -> None:
        """Delete a session."""
        ...

    @abstractmethod
    async def list_sessions(self, user_id: str) -> list[str]:
        """List all session IDs for a user."""
        ...
