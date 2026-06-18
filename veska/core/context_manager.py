"""
Context Window Manager for Veska.

Three layers:
  Layer 1 - Active Context: Only what agent currently needs (in AI conversation)
  Layer 2 - Summaries: Compressed notes of completed tasks (in memory)
  Layer 3 - Full Storage: Complete code and outputs (on disk, retrieved on demand)

Agents never overflow their context window. Old details are summarized,
full data is stored separately and pulled back only when needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from veska.providers.base import Message


class TaskSummary:
    """Compressed summary of a completed task."""

    def __init__(
        self,
        task_id: str,
        summary: str,
        key_facts: list[str],
        files_created: list[str],
        decisions: list[str],
        connections: list[str],
    ) -> None:
        self.task_id = task_id
        self.summary = summary
        self.key_facts = key_facts
        self.files_created = files_created
        self.decisions = decisions
        self.connections = connections

    def to_context_string(self) -> str:
        """Convert to a compact string for injecting into context."""
        parts = [f"Task '{self.task_id}': {self.summary}"]

        if self.key_facts:
            parts.append(f"  Key facts: {', '.join(self.key_facts)}")
        if self.files_created:
            parts.append(f"  Files: {', '.join(self.files_created)}")
        if self.decisions:
            parts.append(f"  Decisions: {', '.join(self.decisions)}")
        if self.connections:
            parts.append(f"  Connections: {', '.join(self.connections)}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "files_created": self.files_created,
            "decisions": self.decisions,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskSummary:
        return cls(**data)


class ContextManager:
    """
    Manages context window for an agent.

    Keeps active context small by:
    - Summarizing completed tasks (Layer 2)
    - Storing full outputs on disk (Layer 3)
    - Pulling specific data back only when needed

    Usage:
        ctx = ContextManager(agent_id="backend_agent", storage_dir="./storage")

        # After completing a task, summarize it
        ctx.complete_task(task_id, summary, key_facts, files, decisions, connections)

        # Get context to inject into conversation
        context = ctx.build_context(current_task="Build auth API")

        # Need full details from an earlier task? Retrieve it
        full_data = ctx.retrieve_full("task_003")
    """

    def __init__(
        self,
        agent_id: str,
        storage_dir: Optional[str] = None,
        max_summaries_in_context: int = 20,
    ) -> None:
        self.agent_id = agent_id
        self.max_summaries = max_summaries_in_context

        # Layer 2: Summaries (in memory)
        self._summaries: list[TaskSummary] = []

        # Layer 3: Full storage (on disk)
        self._storage_dir: Optional[Path] = None
        if storage_dir:
            self._storage_dir = Path(storage_dir) / agent_id
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    def complete_task(
        self,
        task_id: str,
        summary: str,
        key_facts: list[str] | None = None,
        files_created: list[str] | None = None,
        decisions: list[str] | None = None,
        connections: list[str] | None = None,
        full_output: Optional[str] = None,
    ) -> None:
        """
        Record a completed task.

        Saves summary in memory (Layer 2) and full output to disk (Layer 3).
        """
        # Layer 2: Store summary
        task_summary = TaskSummary(
            task_id=task_id,
            summary=summary,
            key_facts=key_facts or [],
            files_created=files_created or [],
            decisions=decisions or [],
            connections=connections or [],
        )
        self._summaries.append(task_summary)

        # Layer 3: Store full output to disk
        if full_output and self._storage_dir:
            self._save_full(task_id, full_output)

    def build_context(self, current_task: Optional[str] = None) -> str:
        """
        Build context string to inject into the conversation.

        Includes recent summaries + current task info.
        Keeps context small and focused.
        """
        parts = []

        # Include summaries of completed tasks
        if self._summaries:
            parts.append("## What you've done so far:")
            # Only include recent summaries to keep context small
            recent = self._summaries[-self.max_summaries:]
            for s in recent:
                parts.append(s.to_context_string())

        # Current task
        if current_task:
            parts.append(f"\n## Current task:\n{current_task}")

        return "\n".join(parts)

    def retrieve_full(self, task_id: str) -> Optional[str]:
        """
        Retrieve full output of a previous task from disk (Layer 3).

        Use this when agent needs complete details from an earlier task,
        not just the summary.
        """
        if not self._storage_dir:
            return None

        file_path = self._storage_dir / f"{task_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
        return data.get("output")

    def get_summaries(self) -> list[TaskSummary]:
        """Get all task summaries."""
        return list(self._summaries)

    def get_summary_for_task(self, task_id: str) -> Optional[TaskSummary]:
        """Get summary for a specific task."""
        for s in self._summaries:
            if s.task_id == task_id:
                return s
        return None

    def _save_full(self, task_id: str, output: str) -> None:
        """Save full output to disk."""
        if not self._storage_dir:
            return

        file_path = self._storage_dir / f"{task_id}.json"
        with open(file_path, "w") as f:
            json.dump({"task_id": task_id, "output": output}, f)

    # --- Conversation history management ---

    def trim_messages(
        self,
        messages: list[Message],
        max_messages: int = 50,
    ) -> list[Message]:
        """
        Trim conversation history to keep it within limits.

        Keeps the system message and recent messages.
        Old messages are dropped (their info lives in summaries).
        """
        if len(messages) <= max_messages:
            return messages

        # Always keep system message (first one)
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # Keep recent messages
        recent = non_system[-(max_messages - len(system_msgs)):]

        return system_msgs + recent

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "summaries": [s.to_dict() for s in self._summaries],
        }

    @classmethod
    def from_dict(cls, data: dict, storage_dir: Optional[str] = None) -> ContextManager:
        ctx = cls(agent_id=data["agent_id"], storage_dir=storage_dir)
        ctx._summaries = [
            TaskSummary.from_dict(s) for s in data.get("summaries", [])
        ]
        return ctx
