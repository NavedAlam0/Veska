"""
Tool Permissions for Veska.

Controls which agent can use which tools.
Each agent gets only the tools it needs.
"""

from __future__ import annotations

from typing import Optional

from veska.tools.base import Tool
from veska.tools.registry import ToolRegistry


class ToolPermissions:
    """
    Per-agent tool access control.

    Usage:
        perms = ToolPermissions(registry)

        # Set what tools each agent can use
        perms.set("backend_agent", ["create_file", "read_file", "run_python", "run_command"])
        perms.set("frontend_agent", ["create_file", "read_file", "run_node"])

        # Get tools for an agent
        tools = perms.get_tools("backend_agent")

        # Check access
        perms.can_use("backend_agent", "run_python")  # True
        perms.can_use("frontend_agent", "run_python")  # False
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._permissions: dict[str, list[str]] = {}

    def set(self, agent_name: str, tool_names: list[str]) -> None:
        """Set which tools an agent can use."""
        self._permissions[agent_name] = tool_names

    def grant(self, agent_name: str, tool_name: str) -> None:
        """Grant an agent access to a specific tool."""
        if agent_name not in self._permissions:
            self._permissions[agent_name] = []
        if tool_name not in self._permissions[agent_name]:
            self._permissions[agent_name].append(tool_name)

    def revoke(self, agent_name: str, tool_name: str) -> None:
        """Revoke an agent's access to a specific tool."""
        if agent_name in self._permissions:
            if tool_name in self._permissions[agent_name]:
                self._permissions[agent_name].remove(tool_name)

    def can_use(self, agent_name: str, tool_name: str) -> bool:
        """Check if an agent can use a tool."""
        if agent_name not in self._permissions:
            # No permissions set = access to all tools
            return True
        return tool_name in self._permissions[agent_name]

    def get_tools(self, agent_name: str) -> list[Tool]:
        """Get the tools an agent is allowed to use."""
        if agent_name not in self._permissions:
            # No permissions set = all tools
            return self._registry.get_all()
        return self._registry.get_for_agent(self._permissions[agent_name])

    def get_tool_names(self, agent_name: str) -> list[str]:
        """Get the tool names an agent is allowed to use."""
        if agent_name not in self._permissions:
            return self._registry.get_names()
        return list(self._permissions[agent_name])
