"""
Base Tool class - unified system for all tools.

Pre-built and custom tools use the same class.
Agent sees one flat list, doesn't know the difference.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Definition of a single tool parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class ToolResult(BaseModel):
    """Result returned after a tool executes."""

    success: bool
    output: Any = None
    error: Optional[str] = None


class Tool(BaseModel):
    """
    Unified tool definition.

    Same class for pre-built tools and custom tools.
    Agent sees all tools the same way.

    Usage:
        # Pre-built (framework creates these)
        file_tool = Tool(
            name="create_file",
            description="Create a new file",
            when_to_use="When you need to create a new code file",
            parameters=[ToolParameter(name="path", description="File path")],
            function=create_file_func,
        )

        # Custom (user creates these the exact same way)
        sms_tool = Tool(
            name="send_sms",
            description="Send an SMS message",
            when_to_use="When task requires SMS notifications",
            parameters=[ToolParameter(name="phone", description="Phone number")],
            function=user_sms_func,
        )
    """

    name: str = Field(description="Unique name for the tool")
    description: str = Field(description="What this tool does")
    when_to_use: str = Field(
        default="", description="Instructions for when the agent should use this tool"
    )
    parameters: list[ToolParameter] = Field(
        default_factory=list, description="Parameters this tool accepts"
    )
    function: Optional[Callable] = Field(
        default=None, exclude=True, description="The actual function to execute"
    )

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        if self.function is None:
            return ToolResult(success=False, error=f"Tool '{self.name}' has no function")

        try:
            # Support both sync and async functions
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def to_schema(self) -> dict:
        """
        Convert tool to a schema dict for AI model consumption.
        This format is used in the system prompt so the agent
        knows about this tool.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_provider_format(self, provider: str = "claude") -> dict:
        """
        Convert tool to the format expected by AI providers.
        Each provider has slightly different tool formats.
        """
        schema = self.to_schema()

        if provider == "claude":
            return {
                "name": schema["name"],
                "description": self._build_description(),
                "input_schema": schema["parameters"],
            }
        elif provider == "openai":
            return {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": self._build_description(),
                    "parameters": schema["parameters"],
                },
            }
        else:
            # Default: use Claude format as fallback
            return {
                "name": schema["name"],
                "description": self._build_description(),
                "input_schema": schema["parameters"],
            }

    def _build_description(self) -> str:
        """Build full description including when_to_use."""
        desc = self.description
        if self.when_to_use:
            desc += f"\n\nWhen to use: {self.when_to_use}"
        return desc
