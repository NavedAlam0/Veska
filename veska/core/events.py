"""
Event System for Veska.

Framework emits events (signals). Any app listens and decides how to display them.
Framework doesn't know or care about the UI.

Events: framework -> app (things happened)
Inputs: app -> framework (user wants something)

Like a car engine sending signals to the dashboard.
Engine doesn't care if dashboard is digital, analog, or a phone app.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events the framework can emit."""

    # Lifecycle
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    RESUMED = "resumed"
    CANCELLED = "cancelled"

    # Checkpoints (pause for user approval)
    CHECKPOINT = "checkpoint"
    CHECKPOINT_APPROVED = "checkpoint_approved"
    CHECKPOINT_REJECTED = "checkpoint_rejected"

    # Progress
    PROGRESS = "progress"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"

    # Agent activity
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_MESSAGE = "agent_message"

    # Task
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Errors & Recovery
    ERROR = "error"
    BUG_DETECTED = "bug_detected"
    DISCUSSION_STARTED = "discussion_started"
    FIX_STARTED = "fix_started"

    # User interaction
    NEED_INPUT = "need_input"
    USER_FEEDBACK = "user_feedback"

    # Thinking (if output is "expose")
    THINKING = "thinking"


class Event(BaseModel):
    """A single event emitted by the framework."""

    type: EventType
    source: str = ""  # who emitted (agent name or "framework")
    message: str = ""
    data: Optional[dict] = None
    timestamp: float = Field(default_factory=time.time)


class CheckpointData(BaseModel):
    """Data for a checkpoint event."""

    checkpoint_id: str
    title: str
    description: str
    requires_approval: bool = True
    details: Optional[dict] = None


class ProgressData(BaseModel):
    """Data for a progress event."""

    completed: int
    total: int
    current_task: str = ""
    percentage: float = 0.0


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[Event], None]


class EventEmitter:
    """
    Event system for Veska framework.

    Framework emits events. Any app (web, CLI, desktop) listens.
    Framework doesn't know about the UI.

    Usage:
        emitter = EventEmitter()

        # App listens to events
        emitter.on(EventType.CHECKPOINT, handle_checkpoint)
        emitter.on(EventType.PROGRESS, handle_progress)
        emitter.on(EventType.ERROR, handle_error)

        # Listen to ALL events
        emitter.on_any(handle_any_event)

        # Framework emits events
        await emitter.emit(Event(
            type=EventType.PROGRESS,
            source="backend_agent",
            message="3 of 5 tasks done",
            data={"completed": 3, "total": 5},
        ))

        # User input (app -> framework)
        emitter.set_input_handler(handle_user_input)
    """

    def __init__(self) -> None:
        # Event handlers per type
        self._handlers: dict[EventType, list[EventHandler]] = {}

        # Catch-all handlers (see every event)
        self._any_handlers: list[EventHandler] = []

        # Sync handlers (for simpler use cases)
        self._sync_handlers: dict[EventType, list[SyncEventHandler]] = {}
        self._sync_any_handlers: list[SyncEventHandler] = []

        # Event history
        self._history: list[Event] = []

        # Pending checkpoints waiting for user response
        self._pending_checkpoints: dict[str, asyncio.Future] = {}

        # User input handler
        self._input_handler: Optional[Callable] = None

    # --- Register handlers ---

    def on(self, event_type: EventType, handler: EventHandler | SyncEventHandler) -> None:
        """Register a handler for a specific event type."""
        if asyncio.iscoroutinefunction(handler):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
        else:
            if event_type not in self._sync_handlers:
                self._sync_handlers[event_type] = []
            self._sync_handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler | SyncEventHandler) -> None:
        """Register a handler that receives ALL events."""
        if asyncio.iscoroutinefunction(handler):
            self._any_handlers.append(handler)
        else:
            self._sync_any_handlers.append(handler)

    def off(self, event_type: EventType, handler: EventHandler | SyncEventHandler) -> None:
        """Remove a handler."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
        if event_type in self._sync_handlers and handler in self._sync_handlers[event_type]:
            self._sync_handlers[event_type].remove(handler)

    # --- Emit events ---

    async def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        self._history.append(event)

        # Async handlers for this event type
        for handler in self._handlers.get(event.type, []):
            try:
                await handler(event)
            except Exception:
                pass

        # Sync handlers for this event type
        for handler in self._sync_handlers.get(event.type, []):
            try:
                handler(event)
            except Exception:
                pass

        # Catch-all async handlers
        for handler in self._any_handlers:
            try:
                await handler(event)
            except Exception:
                pass

        # Catch-all sync handlers
        for handler in self._sync_any_handlers:
            try:
                handler(event)
            except Exception:
                pass

    # --- Convenience emit methods ---

    async def emit_progress(
        self,
        source: str,
        completed: int,
        total: int,
        current_task: str = "",
    ) -> None:
        """Emit a progress event."""
        pct = (completed / total * 100) if total > 0 else 0
        await self.emit(Event(
            type=EventType.PROGRESS,
            source=source,
            message=f"{completed}/{total} tasks done ({pct:.0f}%)",
            data={
                "completed": completed,
                "total": total,
                "current_task": current_task,
                "percentage": pct,
            },
        ))

    async def emit_error(self, source: str, error: str) -> None:
        """Emit an error event."""
        await self.emit(Event(
            type=EventType.ERROR,
            source=source,
            message=error,
        ))

    async def emit_agent_status(
        self, agent_name: str, event_type: EventType, message: str = ""
    ) -> None:
        """Emit an agent status event."""
        await self.emit(Event(
            type=event_type,
            source=agent_name,
            message=message,
        ))

    # --- Checkpoints (pause for user approval) ---

    async def checkpoint(
        self,
        checkpoint_id: str,
        title: str,
        description: str,
        details: Optional[dict] = None,
    ) -> dict:
        """
        Emit a checkpoint and wait for user response.

        Returns the user's response (approve/reject + feedback).
        """
        # Create a future that will be resolved when user responds
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_checkpoints[checkpoint_id] = future

        # Emit the checkpoint event
        await self.emit(Event(
            type=EventType.CHECKPOINT,
            source="framework",
            message=title,
            data={
                "checkpoint_id": checkpoint_id,
                "title": title,
                "description": description,
                "details": details,
            },
        ))

        # Wait for user response
        response = await future
        del self._pending_checkpoints[checkpoint_id]
        return response

    def approve_checkpoint(self, checkpoint_id: str, feedback: str = "") -> None:
        """User approves a checkpoint (called by the app)."""
        future = self._pending_checkpoints.get(checkpoint_id)
        if future and not future.done():
            future.set_result({
                "approved": True,
                "feedback": feedback,
            })

    def reject_checkpoint(self, checkpoint_id: str, feedback: str = "") -> None:
        """User rejects a checkpoint (called by the app)."""
        future = self._pending_checkpoints.get(checkpoint_id)
        if future and not future.done():
            future.set_result({
                "approved": False,
                "feedback": feedback,
            })

    # --- User input ---

    async def request_input(self, prompt: str, source: str = "framework") -> None:
        """Request input from the user."""
        await self.emit(Event(
            type=EventType.NEED_INPUT,
            source=source,
            message=prompt,
        ))

    async def send_feedback(self, feedback: str) -> None:
        """User sends feedback (called by the app)."""
        await self.emit(Event(
            type=EventType.USER_FEEDBACK,
            source="user",
            message=feedback,
        ))

    # --- Pause / Resume / Cancel ---

    async def pause(self) -> None:
        """User pauses the framework."""
        await self.emit(Event(
            type=EventType.PAUSED,
            source="user",
            message="Paused by user",
        ))

    async def resume(self) -> None:
        """User resumes the framework."""
        await self.emit(Event(
            type=EventType.RESUMED,
            source="user",
            message="Resumed by user",
        ))

    async def cancel(self) -> None:
        """User cancels the operation."""
        await self.emit(Event(
            type=EventType.CANCELLED,
            source="user",
            message="Cancelled by user",
        ))

    # --- Query ---

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 50,
    ) -> list[Event]:
        """Get event history with optional filters."""
        events = self._history

        if event_type:
            events = [e for e in events if e.type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        return events[-limit:]

    @property
    def stats(self) -> dict:
        """Get event system stats."""
        return {
            "total_events": len(self._history),
            "handlers_registered": sum(
                len(h) for h in self._handlers.values()
            ) + len(self._any_handlers),
            "pending_checkpoints": len(self._pending_checkpoints),
        }
