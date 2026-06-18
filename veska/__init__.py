"""
Veska - A multi-agent framework built from scratch.

General purpose framework for building multi-agent systems.
Supports Claude API and OpenAI API with per-agent configuration.
"""

__version__ = "0.1.0"

# Auto-load .env file on import (searches cwd and parent dirs)
from veska.core.env import load_env as _load_env
_load_env()

# Core
from veska.core.agent import Agent, AgentConfig, AgentResult
from veska.core.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorResult
from veska.core.message_bus import MessageBus, BusMessage, MessageType
from veska.core.events import EventEmitter, EventType, Event
from veska.core.memory import AgentMemory, SharedMemory
from veska.core.task_planner import TaskPlanner, Task, TaskStatus
from veska.core.context_manager import ContextManager
from veska.core.prompt_manager import PromptManager
from veska.core.thinking import ThinkingHandler
from veska.recovery.error_recovery import ErrorDetector, DiscussionRoom, FixCoordinator, FixReport

# Providers
from veska.providers.base import BaseProvider, Message, ProviderResponse, StreamEvent
from veska.providers.claude_provider import ClaudeProvider
from veska.providers.openai_provider import OpenAIProvider

# Tools
from veska.tools.base import Tool, ToolParameter, ToolResult
from veska.tools.registry import ToolRegistry
from veska.tools.permissions import ToolPermissions

# Security
from veska.security.sandbox import Sandbox
from veska.security.command_guard import CommandGuard
from veska.security.code_scanner import CodeScanner

# Sessions
from veska.sessions.store import SessionStore
from veska.sessions.file_store import FileSessionStore
from veska.sessions.sqlite_store import SQLiteSessionStore

# Media (multi-modal)
from veska.media.types import Image, PDF, Audio

# Cache
from veska.cache.store import CacheStore
from veska.cache.memory_cache import InMemoryCache
from veska.cache.file_cache import FileCache

# Memory (persistent)
from veska.memory.store import Memory, MemoryStore
from veska.memory.file_store import FileMemoryStore
from veska.memory.sqlite_store import SQLiteMemoryStore
from veska.memory.migrate import migrate_memory

# Optional systems
from veska.logging.logger import Logger, LogLevel
from veska.tracking.cost_tracker import CostTracker
from veska.recovery.recovery import RecoveryManager
from veska.core.mcp_connector import MCPConnector, MCPServer

__all__ = [
    # Core
    "Agent", "AgentConfig", "AgentResult",
    "Orchestrator", "OrchestratorConfig", "OrchestratorResult",
    "MessageBus", "BusMessage", "MessageType",
    "EventEmitter", "EventType", "Event",
    "AgentMemory", "SharedMemory",
    "TaskPlanner", "Task", "TaskStatus",
    "ContextManager",
    "PromptManager",
    "ThinkingHandler",
    "ErrorDetector", "DiscussionRoom", "FixCoordinator", "FixReport",
    # Providers
    "BaseProvider", "Message", "ProviderResponse", "StreamEvent",
    "ClaudeProvider", "OpenAIProvider",
    # Tools
    "Tool", "ToolParameter", "ToolResult",
    "ToolRegistry", "ToolPermissions",
    # Sessions
    "SessionStore", "FileSessionStore", "SQLiteSessionStore",
    # Media
    "Image", "PDF", "Audio",
    # Cache
    "CacheStore", "InMemoryCache", "FileCache",
    # Memory
    "Memory", "MemoryStore", "FileMemoryStore", "SQLiteMemoryStore", "migrate_memory",
    # Security
    "Sandbox", "CommandGuard", "CodeScanner",
    # Optional
    "Logger", "LogLevel",
    "CostTracker",
    "RecoveryManager",
    "MCPConnector", "MCPServer",
]
