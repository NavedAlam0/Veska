"""
MemoryStore interface.

All persistent memory backends implement these 5 methods.
Community can build Pinecone, Qdrant, Redis, FAISS — just implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field
import time


class Memory(BaseModel):
    """A single persisted memory entry."""

    key: str
    value: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class MemoryStore(ABC):
    """
    Abstract base for all memory backends.

    Implement these 5 methods to create a custom backend.
    """

    @abstractmethod
    async def save(self, key: str, value: str, metadata: Optional[dict] = None) -> None:
        """Save a memory entry."""
        ...

    @abstractmethod
    async def load(self, key: str) -> Optional[Memory]:
        """Load a single memory by key."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[Memory]:
        """Search memories. Exact match for simple stores, semantic for vector stores."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a memory by key."""
        ...

    @abstractmethod
    async def list_all(self) -> list[Memory]:
        """List all stored memories."""
        ...
