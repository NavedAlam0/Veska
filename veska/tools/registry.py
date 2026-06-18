"""
Tool Registry - one list for all tools.

Pre-built, custom, and MCP tools all registered here.
Agent sees one flat list. Doesn't know which type each tool is.
"""

from __future__ import annotations

from typing import Optional

from veska.tools.base import Tool


# Pre-built tool names that the framework provides
PREBUILT_TOOLS = {
    "file_manager",
    "code_runner",
    "project_tools",
    "web_tools",
    "database_tools",
    "api_tools",
}


class ToolRegistry:
    """
    Central registry for all tools.

    Usage:
        registry = ToolRegistry()

        # Register pre-built tools by name
        registry.register("file_manager")
        registry.register("code_runner")

        # Register custom tools (same method)
        registry.register(my_custom_tool)

        # Get all tools
        tools = registry.get_all()

        # Get specific tool
        tool = registry.get("file_manager")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: str | Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Either a tool name string (for pre-built) or a Tool instance.
        """
        if isinstance(tool, str):
            self._register_prebuilt(tool)
        elif isinstance(tool, Tool):
            self._register_tool(tool)
        else:
            raise TypeError(f"Expected str or Tool, got {type(tool)}")

    def _register_prebuilt(self, name: str) -> None:
        """Load and register a pre-built tool by name."""
        if name not in PREBUILT_TOOLS:
            raise ValueError(
                f"Unknown pre-built tool: '{name}'. "
                f"Available: {', '.join(sorted(PREBUILT_TOOLS))}"
            )
        # Lazy load pre-built tools to avoid circular imports
        tools = _load_prebuilt(name)
        for tool in tools:
            self._register_tool(tool)

    def _register_tool(self, tool: Tool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered")
        del self._tools[name]

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_for_agent(self, allowed_tools: list[str]) -> list[Tool]:
        """
        Get tools filtered by what an agent is allowed to use.

        Args:
            allowed_tools: List of tool names this agent can use.

        Returns:
            List of Tool instances the agent has access to.
        """
        return [
            tool
            for name, tool in self._tools.items()
            if name in allowed_tools
        ]

    def to_provider_format(
        self, provider: str = "claude", allowed_tools: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Convert tools to provider format for API calls.

        Args:
            provider: "claude" or "openai"
            allowed_tools: If set, only include these tools.

        Returns:
            List of tool definitions in provider format.
        """
        tools = (
            self.get_for_agent(allowed_tools)
            if allowed_tools
            else self.get_all()
        )
        return [tool.to_provider_format(provider) for tool in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def _load_prebuilt(name: str) -> list[Tool]:
    """
    Load pre-built tools by name.

    Returns a list because one tool group (like "file_manager")
    can contain multiple individual tools.
    """
    # Will be implemented when we build pre-built tools in Step 13
    # For now, return empty list as placeholder
    loaders = {
        "file_manager": _load_file_manager_tools,
        "code_runner": _load_code_runner_tools,
        "project_tools": _load_project_tools,
        "web_tools": _load_web_tools,
        "database_tools": _load_database_tools,
        "api_tools": _load_api_tools,
    }

    loader = loaders.get(name)
    if loader is None:
        return []
    return loader()


def _load_file_manager_tools() -> list[Tool]:
    from veska.tools.file_manager import get_file_manager_tools
    return get_file_manager_tools()


def _load_code_runner_tools() -> list[Tool]:
    from veska.tools.code_runner import get_code_runner_tools
    return get_code_runner_tools()


def _load_project_tools() -> list[Tool]:
    from veska.tools.project_tools import get_project_tools
    return get_project_tools()


# Optional tools - placeholders until implemented
def _load_web_tools() -> list[Tool]:
    return []


def _load_database_tools() -> list[Tool]:
    return []


def _load_api_tools() -> list[Tool]:
    return []
