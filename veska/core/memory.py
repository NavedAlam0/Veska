"""
Memory System for Veska.

Each agent has private memory. Orchestrator can read any agent's memory
and share one agent's memory with another.

No restarting from scratch - if something breaks, we find exactly
what broke and fix only that part.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str
    value: Any
    category: str = "general"  # decisions, files, tasks, errors, connections
    timestamp: float = Field(default_factory=time.time)


class AgentMemory:
    """
    Private memory for a single agent.

    Stores decisions, files created, tasks done, errors hit,
    and connections to other agents' work.

    Usage:
        memory = AgentMemory(agent_id="backend_agent")
        memory.add_decision("Chose PostgreSQL over MongoDB")
        memory.add_file("server.py", "Main server entry point")
        memory.add_task("auth_api", "Created JWT authentication")
        memory.add_error("Port 3000 conflict, switched to 3001")

        # Get summary for context window
        summary = memory.get_summary()

        # Get full memory for sharing
        full = memory.get_all()
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._entries: list[MemoryEntry] = []
        self._current_state: str = "idle"

    # --- Add memories by category ---

    def add_decision(self, decision: str) -> None:
        """Record a decision the agent made."""
        self._add("decision", decision, "decisions")

    def add_file(self, path: str, description: str = "") -> None:
        """Record a file the agent created or modified."""
        self._add(path, description, "files")

    def add_task(self, task_id: str, result: str) -> None:
        """Record a completed task."""
        self._add(task_id, result, "tasks")

    def add_error(self, error: str) -> None:
        """Record an error the agent encountered."""
        self._add("error", error, "errors")

    def add_connection(self, connection: str) -> None:
        """Record a dependency or connection to other agents' work."""
        self._add("connection", connection, "connections")

    def add(self, key: str, value: Any, category: str = "general") -> None:
        """Add a generic memory entry."""
        self._add(key, value, category)

    def _add(self, key: str, value: Any, category: str) -> None:
        """Internal method to add a memory entry."""
        self._entries.append(
            MemoryEntry(key=key, value=value, category=category)
        )

    # --- Update state ---

    def set_state(self, state: str) -> None:
        """Update the agent's current state."""
        self._current_state = state

    @property
    def current_state(self) -> str:
        return self._current_state

    # --- Retrieve memories ---

    def get_by_category(self, category: str) -> list[MemoryEntry]:
        """Get all entries in a category."""
        return [e for e in self._entries if e.category == category]

    def get_decisions(self) -> list[str]:
        """Get all decisions."""
        return [e.value for e in self.get_by_category("decisions")]

    def get_files(self) -> dict[str, str]:
        """Get all files as {path: description}."""
        return {e.key: e.value for e in self.get_by_category("files")}

    def get_tasks(self) -> dict[str, str]:
        """Get all completed tasks as {task_id: result}."""
        return {e.key: e.value for e in self.get_by_category("tasks")}

    def get_errors(self) -> list[str]:
        """Get all errors."""
        return [e.value for e in self.get_by_category("errors")]

    def get_connections(self) -> list[str]:
        """Get all connections."""
        return [e.value for e in self.get_by_category("connections")]

    def get_all(self) -> list[MemoryEntry]:
        """Get all memory entries."""
        return list(self._entries)

    def get_recent(self, count: int = 10) -> list[MemoryEntry]:
        """Get the most recent N entries."""
        return self._entries[-count:]

    # --- Summary for context window ---

    def get_summary(self) -> str:
        """
        Get a compressed summary of this agent's memory.
        Used for context window management - keeps context small.
        """
        parts = [f"Agent: {self.agent_id}", f"State: {self._current_state}"]

        decisions = self.get_decisions()
        if decisions:
            parts.append(f"Decisions: {', '.join(decisions[-5:])}")

        files = self.get_files()
        if files:
            file_list = list(files.keys())[-10:]
            parts.append(f"Files created: {', '.join(file_list)}")

        tasks = self.get_tasks()
        if tasks:
            task_list = [f"{k}: {v}" for k, v in list(tasks.items())[-5:]]
            parts.append(f"Tasks done: {'; '.join(task_list)}")

        errors = self.get_errors()
        if errors:
            parts.append(f"Errors hit: {'; '.join(errors[-3:])}")

        connections = self.get_connections()
        if connections:
            parts.append(f"Connections: {'; '.join(connections[-5:])}")

        return "\n".join(parts)

    # --- Serialization for save points ---

    def to_dict(self) -> dict:
        """Serialize memory for saving (crash recovery)."""
        return {
            "agent_id": self.agent_id,
            "current_state": self._current_state,
            "entries": [e.model_dump() for e in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentMemory:
        """Restore memory from saved data."""
        memory = cls(agent_id=data["agent_id"])
        memory._current_state = data.get("current_state", "idle")
        memory._entries = [
            MemoryEntry(**entry) for entry in data.get("entries", [])
        ]
        return memory

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries.clear()
        self._current_state = "idle"

    def __len__(self) -> int:
        return len(self._entries)


class SharedMemory:
    """
    Shared memory pool managed by the Orchestrator.

    Orchestrator controls which agent can see which memory.
    Agents don't access this directly - Orchestrator mediates.

    Usage:
        shared = SharedMemory()

        # Store agent memory
        shared.store(agent_memory)

        # Get one agent's memory
        memory = shared.get("backend_agent")

        # Share specific info between agents
        shared.share("backend_agent", "frontend_agent", category="files")

        # Get summary of all agents
        overview = shared.get_overview()
    """

    def __init__(self) -> None:
        self._memories: dict[str, AgentMemory] = {}

    def store(self, memory: AgentMemory) -> None:
        """Store or update an agent's memory."""
        self._memories[memory.agent_id] = memory

    def get(self, agent_id: str) -> Optional[AgentMemory]:
        """Get an agent's memory."""
        return self._memories.get(agent_id)

    def get_summary(self, agent_id: str) -> Optional[str]:
        """Get an agent's memory summary."""
        memory = self._memories.get(agent_id)
        return memory.get_summary() if memory else None

    def share(
        self,
        from_agent: str,
        to_agent: str,
        category: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        Get entries from one agent's memory to share with another.

        Args:
            from_agent: Agent whose memory to read.
            to_agent: Agent who will receive the info.
            category: Optional filter - only share this category.

        Returns:
            List of memory entries to share.
        """
        source = self._memories.get(from_agent)
        if source is None:
            return []

        if category:
            return source.get_by_category(category)
        return source.get_all()

    def get_overview(self) -> str:
        """Get a brief overview of all agents' states."""
        lines = []
        for agent_id, memory in self._memories.items():
            lines.append(
                f"- {agent_id}: {memory.current_state} "
                f"({len(memory)} memories)"
            )
        return "\n".join(lines)

    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs with stored memory."""
        return list(self._memories.keys())

    def to_dict(self) -> dict:
        """Serialize all memories for saving."""
        return {
            agent_id: memory.to_dict()
            for agent_id, memory in self._memories.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> SharedMemory:
        """Restore from saved data."""
        shared = cls()
        for agent_id, memory_data in data.items():
            shared._memories[agent_id] = AgentMemory.from_dict(memory_data)
        return shared
