"""
Task Planner for Veska.

Manages tasks with dependency graph and hybrid parallel/sequential execution.

Rules:
  - No dependencies -> start immediately (parallel)
  - Has dependencies -> wait for those to finish first (sequential)
  - Same level, no connection -> run parallel

The Orchestrator uses this to plan and track all work.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a task in the pipeline."""

    WAITING = "waiting"       # waiting for dependencies
    READY = "ready"           # dependencies met, can start
    RUNNING = "running"       # agent is working on it
    DONE = "done"             # completed successfully
    FAILED = "failed"         # failed
    CANCELLED = "cancelled"   # cancelled by user or orchestrator
    RETRYING = "retrying"     # being retried after failure


class Task(BaseModel):
    """
    A single task in the execution plan.

    Each task is assigned to an agent and may depend on other tasks.
    """

    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    agent: str  # which agent handles this
    depends_on: list[str] = Field(default_factory=list)  # task IDs
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0  # higher = more important
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 2

    @property
    def duration(self) -> Optional[float]:
        """How long the task took (if completed)."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a final state."""
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def can_retry(self) -> bool:
        """Whether the task can be retried."""
        return self.status == TaskStatus.FAILED and self.retries < self.max_retries


class TaskPlan(BaseModel):
    """
    A collection of tasks forming an execution plan.
    Tracks phases for checkpoint support.
    """

    id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    phases: list[Phase] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class Phase(BaseModel):
    """
    A group of related tasks that form a logical phase.
    Used for checkpoints - user can approve after each phase.
    """

    name: str
    description: str = ""
    task_ids: list[str] = Field(default_factory=list)


class TaskPlanner:
    """
    Manages the task dependency graph and execution order.

    Builds a dependency graph, determines which tasks can run in parallel,
    and tracks progress through the plan.

    Usage:
        planner = TaskPlanner()

        # Add tasks with dependencies
        planner.add_task(Task(
            id="design",
            name="Design architecture",
            agent="architect",
        ))
        planner.add_task(Task(
            id="backend",
            name="Build backend",
            agent="backend_agent",
            depends_on=["design"],
        ))
        planner.add_task(Task(
            id="frontend",
            name="Build frontend",
            agent="frontend_agent",
            depends_on=["design"],
        ))
        planner.add_task(Task(
            id="qa",
            name="Test everything",
            agent="qa_agent",
            depends_on=["backend", "frontend"],
        ))

        # Get tasks ready to run (no unmet dependencies)
        ready = planner.get_ready_tasks()
        # Returns: [design]  (no dependencies)

        # Mark task as done
        planner.complete_task("design", "Architecture plan ready")

        # Now check again
        ready = planner.get_ready_tasks()
        # Returns: [backend, frontend]  (both can run in PARALLEL)
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._phases: list[Phase] = []
        self._plan_name: str = ""
        self._plan_description: str = ""

    # --- Task management ---

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        # Validate dependencies exist (if already added)
        for dep_id in task.depends_on:
            if dep_id not in self._tasks:
                # Dependency might be added later, that's ok
                pass

        self._tasks[task.id] = task

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add multiple tasks at once."""
        for task in tasks:
            self.add_task(task)

    def remove_task(self, task_id: str) -> None:
        """Remove a task and clean up dependencies."""
        if task_id not in self._tasks:
            return

        del self._tasks[task_id]

        # Remove this task from other tasks' dependencies
        for task in self._tasks.values():
            if task_id in task.depends_on:
                task.depends_on.remove(task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks."""
        return list(self._tasks.values())

    # --- Dependency graph queries ---

    def get_ready_tasks(self) -> list[Task]:
        """
        Get all tasks that are ready to run.

        A task is ready if:
          - Status is WAITING or READY
          - All dependencies are DONE
        """
        ready = []
        for task in self._tasks.values():
            if task.status not in (TaskStatus.WAITING, TaskStatus.READY):
                continue

            if self._dependencies_met(task):
                task.status = TaskStatus.READY
                ready.append(task)

        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies of a task are completed."""
        for dep_id in task.depends_on:
            dep = self._tasks.get(dep_id)
            if dep is None:
                # Dependency doesn't exist - treat as met
                continue
            if dep.status != TaskStatus.DONE:
                return False
        return True

    def get_dependents(self, task_id: str) -> list[Task]:
        """Get all tasks that depend on the given task."""
        return [
            t for t in self._tasks.values()
            if task_id in t.depends_on
        ]

    def get_dependencies(self, task_id: str) -> list[Task]:
        """Get all tasks that the given task depends on."""
        task = self._tasks.get(task_id)
        if not task:
            return []
        return [
            self._tasks[dep_id]
            for dep_id in task.depends_on
            if dep_id in self._tasks
        ]

    # --- Task state transitions ---

    def start_task(self, task_id: str) -> Optional[Task]:
        """Mark a task as running."""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.READY, TaskStatus.WAITING, TaskStatus.RETRYING):
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            return task
        return None

    def complete_task(self, task_id: str, result: str = "") -> Optional[Task]:
        """Mark a task as done with its result."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.DONE
            task.result = result
            task.completed_at = time.time()
            return task
        return None

    def fail_task(self, task_id: str, error: str = "") -> Optional[Task]:
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            return task
        return None

    def retry_task(self, task_id: str) -> Optional[Task]:
        """Retry a failed task."""
        task = self._tasks.get(task_id)
        if task and task.can_retry:
            task.status = TaskStatus.RETRYING
            task.retries += 1
            task.error = None
            task.result = None
            task.started_at = None
            task.completed_at = None
            return task
        return None

    def cancel_task(self, task_id: str) -> Optional[Task]:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if task and not task.is_terminal:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            return task
        return None

    # --- Phase management ---

    def add_phase(self, name: str, task_ids: list[str], description: str = "") -> Phase:
        """Group tasks into a phase for checkpoint support."""
        phase = Phase(name=name, description=description, task_ids=task_ids)
        self._phases.append(phase)
        return phase

    def get_current_phase(self) -> Optional[Phase]:
        """Get the current active phase (first incomplete phase)."""
        for phase in self._phases:
            if not self._phase_complete(phase):
                return phase
        return None

    def is_phase_complete(self, phase_name: str) -> bool:
        """Check if a phase is complete."""
        for phase in self._phases:
            if phase.name == phase_name:
                return self._phase_complete(phase)
        return False

    def _phase_complete(self, phase: Phase) -> bool:
        """Check if all tasks in a phase are done."""
        for task_id in phase.task_ids:
            task = self._tasks.get(task_id)
            if task and task.status != TaskStatus.DONE:
                return False
        return True

    # --- Progress tracking ---

    @property
    def progress(self) -> dict:
        """Get overall progress."""
        total = len(self._tasks)
        if total == 0:
            return {"completed": 0, "total": 0, "percentage": 0.0}

        done = sum(1 for t in self._tasks.values() if t.status == TaskStatus.DONE)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        waiting = sum(
            1 for t in self._tasks.values()
            if t.status in (TaskStatus.WAITING, TaskStatus.READY)
        )

        return {
            "completed": done,
            "running": running,
            "failed": failed,
            "waiting": waiting,
            "total": total,
            "percentage": (done / total) * 100,
        }

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        return all(t.is_terminal for t in self._tasks.values())

    @property
    def has_failures(self) -> bool:
        """Check if any tasks failed."""
        return any(t.status == TaskStatus.FAILED for t in self._tasks.values())

    def get_failed_tasks(self) -> list[Task]:
        """Get all failed tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]

    def get_tasks_by_agent(self, agent: str) -> list[Task]:
        """Get all tasks assigned to a specific agent."""
        return [t for t in self._tasks.values() if t.agent == agent]

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    # --- Validation ---

    def validate(self) -> list[str]:
        """
        Validate the task plan.

        Returns list of errors (empty if valid).
        """
        errors = []

        # Check for missing dependencies
        for task in self._tasks.values():
            for dep_id in task.depends_on:
                if dep_id not in self._tasks:
                    errors.append(
                        f"Task '{task.id}' depends on '{dep_id}' which doesn't exist"
                    )

        # Check for circular dependencies
        if self._has_circular_deps():
            errors.append("Circular dependency detected in task graph")

        return errors

    def _has_circular_deps(self) -> bool:
        """Detect circular dependencies using DFS."""
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(task_id: str) -> bool:
            if task_id in in_stack:
                return True  # cycle found
            if task_id in visited:
                return False

            visited.add(task_id)
            in_stack.add(task_id)

            task = self._tasks.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    if dfs(dep_id):
                        return True

            in_stack.remove(task_id)
            return False

        for task_id in self._tasks:
            if dfs(task_id):
                return True
        return False

    # --- Execution order ---

    def get_execution_order(self) -> list[list[str]]:
        """
        Get the execution order as waves.

        Each wave is a list of task IDs that can run in parallel.
        Waves execute sequentially.

        Returns:
            [[task1, task2], [task3], [task4, task5]]
            Wave 1: task1 and task2 in parallel
            Wave 2: task3 (waits for wave 1)
            Wave 3: task4 and task5 in parallel
        """
        remaining = set(self._tasks.keys())
        completed: set[str] = set()
        waves: list[list[str]] = []

        while remaining:
            # Find tasks whose dependencies are all in 'completed'
            wave = []
            for task_id in list(remaining):
                task = self._tasks[task_id]
                deps = set(task.depends_on)
                if deps.issubset(completed):
                    wave.append(task_id)

            if not wave:
                # No tasks can proceed - circular dependency or error
                break

            waves.append(wave)
            for task_id in wave:
                remaining.remove(task_id)
                completed.add(task_id)

        return waves

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize the plan for saving."""
        return {
            "plan_name": self._plan_name,
            "plan_description": self._plan_description,
            "tasks": {
                task_id: task.model_dump()
                for task_id, task in self._tasks.items()
            },
            "phases": [p.model_dump() for p in self._phases],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskPlanner:
        """Restore from saved data."""
        planner = cls()
        planner._plan_name = data.get("plan_name", "")
        planner._plan_description = data.get("plan_description", "")

        for task_data in data.get("tasks", {}).values():
            planner.add_task(Task(**task_data))

        for phase_data in data.get("phases", []):
            planner._phases.append(Phase(**phase_data))

        return planner

    def __len__(self) -> int:
        return len(self._tasks)
