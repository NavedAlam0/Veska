"""
ChromaDB-based memory store.

Vector search — finds memories by meaning, not exact words.
Requires: pip install chromadb
"""

from __future__ import annotations

import time
import json
from typing import Optional

from veska.memory.store import Memory, MemoryStore


class ChromaMemoryStore(MemoryStore):
    """ChromaDB-backed memory store with semantic search."""

    def __init__(self, directory: str, collection_name: str = "veska_memories") -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required for ChromaMemoryStore. "
                "Install it with: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(path=directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def save(self, key: str, value: str, metadata: Optional[dict] = None) -> None:
        ts = time.time()
        doc_id = f"{key}_{ts}"
        meta = metadata or {}
        meta["key"] = key
        meta["timestamp"] = ts

        self._collection.add(
            ids=[doc_id],
            documents=[f"{key}: {value}"],
            metadatas=[meta],
        )

    async def load(self, key: str) -> Optional[Memory]:
        results = self._collection.get(
            where={"key": key},
            limit=1,
        )
        if not results["ids"]:
            return None

        doc = results["documents"][0]
        meta = results["metadatas"][0]
        # Extract value from "key: value" format
        value = doc.split(": ", 1)[1] if ": " in doc else doc

        return Memory(
            key=key,
            value=value,
            metadata={k: v for k, v in meta.items() if k not in ("key", "timestamp")},
            timestamp=meta.get("timestamp", 0),
        )

    async def search(self, query: str, limit: int = 5) -> list[Memory]:
        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                key = meta.get("key", "")
                value = doc.split(": ", 1)[1] if ": " in doc else doc

                memories.append(Memory(
                    key=key,
                    value=value,
                    metadata={k: v for k, v in meta.items() if k not in ("key", "timestamp")},
                    timestamp=meta.get("timestamp", 0),
                ))

        return memories

    async def delete(self, key: str) -> None:
        results = self._collection.get(where={"key": key})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    async def list_all(self) -> list[Memory]:
        results = self._collection.get()
        memories = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            key = meta.get("key", "")
            value = doc.split(": ", 1)[1] if ": " in doc else doc

            memories.append(Memory(
                key=key,
                value=value,
                metadata={k: v for k, v in meta.items() if k not in ("key", "timestamp")},
                timestamp=meta.get("timestamp", 0),
            ))

        return memories
