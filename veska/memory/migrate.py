"""
Memory migration tool.

Move data between any two memory backends.
No data loss — reads everything from source, writes to destination.
"""

from __future__ import annotations

from veska.memory.store import MemoryStore


async def migrate_memory(
    from_store: MemoryStore,
    to_store: MemoryStore,
) -> int:
    """
    Migrate all memories from one store to another.

    Returns the number of memories migrated.
    """
    memories = await from_store.list_all()

    for mem in memories:
        await to_store.save(
            key=mem.key,
            value=mem.value,
            metadata=mem.metadata,
        )

    return len(memories)
