"""
Resume & Crash Recovery for Veska (Optional).

Auto-retry is always on (built into task planner).
Save points are optional — user provides their own database.

Saves the state of the execution so it can be resumed after a crash.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from veska.core.task_planner import TaskPlanner, Task, TaskStatus


class SavePoint:
    """A snapshot of execution state at a point in time."""

    def __init__(
        self,
        plan_data: dict,
        task_states: dict[str, dict],
        agent_memories: dict[str, dict],
        shared_memory: dict,
        metadata: Optional[dict] = None,
    ) -> None:
        self.id = f"sp_{int(time.time())}"
        self.plan_data = plan_data
        self.task_states = task_states
        self.agent_memories = agent_memories
        self.shared_memory = shared_memory
        self.metadata = metadata or {}
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plan_data": self.plan_data,
            "task_states": self.task_states,
            "agent_memories": self.agent_memories,
            "shared_memory": self.shared_memory,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SavePoint:
        sp = cls(
            plan_data=data["plan_data"],
            task_states=data["task_states"],
            agent_memories=data.get("agent_memories", {}),
            shared_memory=data.get("shared_memory", {}),
            metadata=data.get("metadata", {}),
        )
        sp.id = data.get("id", sp.id)
        sp.created_at = data.get("created_at", sp.created_at)
        return sp


class RecoveryManager:
    """
    Manages save points and crash recovery.

    Auto-retry is always on (handled by TaskPlanner).
    Save points are optional — user opts in and provides storage.

    Usage:
        recovery = RecoveryManager(enabled=True)

        # Option 1: File-based storage (simple)
        recovery.set_storage_dir("/path/to/saves")

        # Option 2: Custom callback (user's database)
        recovery.set_save_callback(my_db_save)
        recovery.set_load_callback(my_db_load)

        # Save a checkpoint
        recovery.save(
            plan_data=planner.to_dict(),
            task_states=get_task_states(),
            agent_memories=get_agent_memories(),
            shared_memory=shared_memory.to_dict(),
        )

        # Resume from last save
        save_point = recovery.load_latest()
        if save_point:
            planner = TaskPlanner.from_dict(save_point.plan_data)
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._storage_dir: Optional[Path] = None
        self._save_callback: Optional[Callable[[SavePoint], None]] = None
        self._load_callback: Optional[Callable[[], Optional[SavePoint]]] = None
        self._save_points: list[SavePoint] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable crash recovery."""
        self._enabled = True

    def disable(self) -> None:
        """Disable crash recovery."""
        self._enabled = False

    def set_storage_dir(self, path: str) -> None:
        """Set a directory for file-based save points."""
        self._storage_dir = Path(path)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def set_save_callback(self, callback: Callable[[SavePoint], None]) -> None:
        """Set a custom save callback (e.g., to user's database)."""
        self._save_callback = callback

    def set_load_callback(self, callback: Callable[[], Optional[SavePoint]]) -> None:
        """Set a custom load callback (e.g., from user's database)."""
        self._load_callback = callback

    def save(
        self,
        plan_data: dict,
        task_states: dict[str, dict],
        agent_memories: Optional[dict[str, dict]] = None,
        shared_memory: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[SavePoint]:
        """
        Save the current execution state.

        Returns the SavePoint if recovery is enabled, None otherwise.
        """
        if not self._enabled:
            return None

        save_point = SavePoint(
            plan_data=plan_data,
            task_states=task_states,
            agent_memories=agent_memories or {},
            shared_memory=shared_memory or {},
            metadata=metadata,
        )

        self._save_points.append(save_point)

        # Persist via callback
        if self._save_callback:
            try:
                self._save_callback(save_point)
            except Exception:
                pass

        # Persist to file
        if self._storage_dir:
            self._save_to_file(save_point)

        return save_point

    def load_latest(self) -> Optional[SavePoint]:
        """Load the most recent save point."""
        # Try custom callback first
        if self._load_callback:
            try:
                return self._load_callback()
            except Exception:
                pass

        # Try file storage
        if self._storage_dir:
            return self._load_from_file()

        # Try in-memory
        if self._save_points:
            return self._save_points[-1]

        return None

    def get_save_points(self) -> list[SavePoint]:
        """Get all in-memory save points."""
        return list(self._save_points)

    def clear(self) -> None:
        """Clear all in-memory save points."""
        self._save_points.clear()

    def _save_to_file(self, save_point: SavePoint) -> None:
        """Save to a JSON file."""
        if not self._storage_dir:
            return

        file_path = self._storage_dir / f"{save_point.id}.json"
        with open(file_path, "w") as f:
            json.dump(save_point.to_dict(), f, indent=2)

        # Also save as "latest"
        latest_path = self._storage_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(save_point.to_dict(), f, indent=2)

    def _load_from_file(self) -> Optional[SavePoint]:
        """Load the latest save point from file."""
        if not self._storage_dir:
            return None

        latest_path = self._storage_dir / "latest.json"
        if not latest_path.exists():
            return None

        try:
            with open(latest_path) as f:
                data = json.load(f)
            return SavePoint.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def get_resumable_tasks(planner: TaskPlanner) -> list[Task]:
        """
        Get tasks that need to be re-run after a crash.

        Tasks that were RUNNING when the crash happened need to restart.
        Tasks that were WAITING or READY can proceed normally.
        """
        resumable = []
        for task in planner.get_all_tasks():
            if task.status == TaskStatus.RUNNING:
                # Was running during crash - needs restart
                task.status = TaskStatus.READY
                task.started_at = None
                resumable.append(task)
            elif task.status in (TaskStatus.WAITING, TaskStatus.READY):
                resumable.append(task)
        return resumable
