"""
@tool decorator for Veska.

Turns a plain Python function into a Tool automatically.
Reads the function name, parameters, and types from the signature.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from veska.tools.base import Tool, ToolParameter


# Python type to JSON schema type mapping
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_string(python_type: Any) -> str:
    """Convert a Python type annotation to a JSON schema type string."""
    return TYPE_MAP.get(python_type, "string")


def _make_tool(func: Callable, description: str) -> Callable:
    """Build a Tool from a function and attach it."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    params = []
    for param_name, param in sig.parameters.items():
        param_type = _python_type_to_string(hints.get(param_name, str))
        has_default = param.default is not inspect.Parameter.empty

        params.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=param_name.replace("_", " "),
            required=not has_default,
            default=param.default if has_default else None,
        ))

    tool_obj = Tool(
        name=func.__name__,
        description=description,
        parameters=params,
        function=func,
    )

    func._tool = tool_obj
    return func


def tool(func_or_description=None) -> Callable:
    """
    Decorator that turns a function into a Tool.

    Usage:
        @tool
        def get_weather(city: str):
            return f"Weather in {city}: 72°F, sunny"

        @tool("Custom description")
        def search(query: str):
            return results
    """
    # @tool (no arguments) — func_or_description is the function itself
    if callable(func_or_description):
        desc = func_or_description.__name__.replace("_", " ").capitalize()
        return _make_tool(func_or_description, desc)

    # @tool("description") — func_or_description is the description string
    description = func_or_description or ""

    def decorator(func: Callable) -> Callable:
        desc = description if description else func.__name__.replace("_", " ").capitalize()
        return _make_tool(func, desc)

    return decorator
