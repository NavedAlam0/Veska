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
from veska.providers.base import BaseProvider, Message, ProviderResponse
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
    "BaseProvider", "Message", "ProviderResponse",
    "ClaudeProvider", "OpenAIProvider",
    # Tools
    "Tool", "ToolParameter", "ToolResult",
    "ToolRegistry", "ToolPermissions",
    # Security
    "Sandbox", "CommandGuard", "CodeScanner",
    # Optional
    "Logger", "LogLevel",
    "CostTracker",
    "RecoveryManager",
    "MCPConnector", "MCPServer",
]
