"""
Message Bus for Veska.

Central communication system. Any agent can send a message to any other agent.
The Orchestrator watches all messages and controls routing.

Flexible message passing (Option B):
  - Any agent talks to any agent
  - Orchestrator decides who talks to whom
  - Messages go through central bus
  - Supports request/response/error/broadcast
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can flow through the bus."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    BROADCAST = "broadcast"       # sent to all agents
    TASK_ASSIGN = "task_assign"   # orchestrator assigns work
    TASK_RESULT = "task_result"   # agent reports completion
    FIX_REQUEST = "fix_request"   # coordinated fix needed
    FEEDBACK = "feedback"         # user feedback


class BusMessage(BaseModel):
    """A single message flowing through the bus."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    from_agent: str
    to_agent: str  # use "*" for broadcast
    type: MessageType
    content: str
    data: Optional[dict] = None
    reply_to: Optional[str] = None  # id of message this replies to
    timestamp: float = Field(default_factory=time.time)


# Type alias for message handler callbacks
MessageHandler = Callable[[BusMessage], Coroutine[Any, Any, None]]


class MessageBus:
    """
    Central message bus for agent communication.

    All messages flow through here. Orchestrator watches and controls routing.

    Usage:
        bus = MessageBus()

        # Agents subscribe to receive messages
        bus.subscribe("backend_agent", handle_message)
        bus.subscribe("frontend_agent", handle_message)

        # Send a message
        await bus.send(BusMessage(
            from_agent="frontend_agent",
            to_agent="backend_agent",
            type=MessageType.REQUEST,
            content="What endpoints do you have?",
        ))

        # Broadcast to all
        await bus.broadcast(
            from_agent="orchestrator",
            content="Project plan is ready",
        )

        # Orchestrator watches all messages
        bus.add_watcher(orchestrator_handler)
    """

    def __init__(self) -> None:
        # Agent handlers: agent_name -> callback
        self._subscribers: dict[str, MessageHandler] = {}

        # Watchers see ALL messages (orchestrator uses this)
        self._watchers: list[MessageHandler] = []

        # Message history
        self._history: list[BusMessage] = []

        # Pending messages (when recipient isn't subscribed yet)
        self._pending: dict[str, list[BusMessage]] = {}

        # Stats
        self._sent_count: int = 0
        self._delivered_count: int = 0

    def subscribe(self, agent_name: str, handler: MessageHandler) -> None:
        """
        Subscribe an agent to receive messages.

        Args:
            agent_name: The agent's name.
            handler: Async callback that processes incoming messages.
        """
        self._subscribers[agent_name] = handler

        # Deliver any pending messages
        if agent_name in self._pending:
            pending = self._pending.pop(agent_name)
            for msg in pending:
                asyncio.create_task(self._deliver(msg, handler))

    def unsubscribe(self, agent_name: str) -> None:
        """Remove an agent's subscription."""
        self._subscribers.pop(agent_name, None)

    def add_watcher(self, handler: MessageHandler) -> None:
        """
        Add a watcher that sees ALL messages.
        Used by Orchestrator to monitor communication.
        """
        self._watchers.append(handler)

    def remove_watcher(self, handler: MessageHandler) -> None:
        """Remove a watcher."""
        if handler in self._watchers:
            self._watchers.remove(handler)

    async def send(self, message: BusMessage) -> None:
        """
        Send a message through the bus.

        Routes to the correct agent. If agent isn't subscribed,
        message is queued as pending.
        """
        self._history.append(message)
        self._sent_count += 1

        # Notify all watchers (orchestrator sees everything)
        for watcher in self._watchers:
            try:
                await watcher(message)
            except Exception:
                pass  # watchers shouldn't break the bus

        # Route the message
        if message.to_agent == "*":
            # Broadcast to all subscribers
            await self._broadcast_message(message)
        else:
            # Direct message
            await self._route_message(message)

    async def broadcast(
        self,
        from_agent: str,
        content: str,
        msg_type: MessageType = MessageType.BROADCAST,
        data: Optional[dict] = None,
    ) -> BusMessage:
        """Send a message to all subscribed agents."""
        message = BusMessage(
            from_agent=from_agent,
            to_agent="*",
            type=msg_type,
            content=content,
            data=data,
        )
        await self.send(message)
        return message

    async def request(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        data: Optional[dict] = None,
    ) -> BusMessage:
        """Send a request message to a specific agent."""
        message = BusMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            type=MessageType.REQUEST,
            content=content,
            data=data,
        )
        await self.send(message)
        return message

    async def respond(
        self,
        to_message: BusMessage,
        from_agent: str,
        content: str,
        data: Optional[dict] = None,
    ) -> BusMessage:
        """Send a response to a previous message."""
        message = BusMessage(
            from_agent=from_agent,
            to_agent=to_message.from_agent,
            type=MessageType.RESPONSE,
            content=content,
            data=data,
            reply_to=to_message.id,
        )
        await self.send(message)
        return message

    async def _route_message(self, message: BusMessage) -> None:
        """Route a message to its target agent."""
        handler = self._subscribers.get(message.to_agent)

        if handler:
            await self._deliver(message, handler)
        else:
            # Agent not subscribed yet - queue it
            if message.to_agent not in self._pending:
                self._pending[message.to_agent] = []
            self._pending[message.to_agent].append(message)

    async def _broadcast_message(self, message: BusMessage) -> None:
        """Deliver message to all subscribers except sender."""
        for agent_name, handler in self._subscribers.items():
            if agent_name != message.from_agent:
                await self._deliver(message, handler)

    async def _deliver(self, message: BusMessage, handler: MessageHandler) -> None:
        """Deliver a message to a handler."""
        try:
            await handler(message)
            self._delivered_count += 1
        except Exception:
            pass  # individual handler failures don't break the bus

    # --- Query methods ---

    def get_history(
        self,
        agent: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        limit: int = 50,
    ) -> list[BusMessage]:
        """
        Get message history with optional filters.

        Args:
            agent: Filter by sender or receiver.
            msg_type: Filter by message type.
            limit: Max messages to return.
        """
        messages = self._history

        if agent:
            messages = [
                m for m in messages
                if m.from_agent == agent or m.to_agent == agent
            ]

        if msg_type:
            messages = [m for m in messages if m.type == msg_type]

        return messages[-limit:]

    def get_conversation(
        self, agent_a: str, agent_b: str
    ) -> list[BusMessage]:
        """Get all messages between two agents."""
        return [
            m for m in self._history
            if (m.from_agent == agent_a and m.to_agent == agent_b)
            or (m.from_agent == agent_b and m.to_agent == agent_a)
        ]

    def get_pending_count(self) -> int:
        """Get total number of pending (undelivered) messages."""
        return sum(len(msgs) for msgs in self._pending.values())

    @property
    def stats(self) -> dict:
        """Get bus statistics."""
        return {
            "sent": self._sent_count,
            "delivered": self._delivered_count,
            "pending": self.get_pending_count(),
            "subscribers": len(self._subscribers),
            "watchers": len(self._watchers),
            "history_size": len(self._history),
        }

    def clear_history(self) -> None:
        """Clear message history."""
        self._history.clear()
        self._sent_count = 0
        self._delivered_count = 0
