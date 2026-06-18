"""
Agent delegation tool.

Allows agents to hand off sub-tasks to other agents mid-execution.
Delegation is just a tool — agent calls it like any other tool.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Optional

from veska.tools.base import Tool, ToolParameter, ToolResult


MAX_DELEGATION_DEPTH = 3
DEFAULT_TIMEOUT = 300  # 5 minutes


def create_delegation_tool(
    agent_directory: dict[str, str],
    run_delegate: Callable[..., Coroutine[Any, Any, Any]],
    self_name: str,
    current_depth: int = 0,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tool:
    """
    Create a delegate_task tool for an agent.

    Args:
        agent_directory: {agent_name: description} of available agents.
        run_delegate: async function(agent_name, task) -> str that runs the delegation.
        self_name: Name of the agent receiving this tool (to prevent self-delegation).
        current_depth: Current delegation depth (0 = top level).
        timeout: Max seconds to wait for delegated agent.
    """
    # Build directory string for the tool description
    directory_lines = []
    for name, desc in agent_directory.items():
        if name != self_name:
            directory_lines.append(f"  - {name}: {desc}")

    directory_text = "\n".join(directory_lines) if directory_lines else "  (no other agents available)"

    async def delegate_task(agent: str, task: str) -> str:
        # Guard: can't call yourself
        if agent == self_name:
            return f"Error: Cannot delegate to yourself ({self_name})."

        # Guard: max depth
        if current_depth >= MAX_DELEGATION_DEPTH:
            return f"Error: Maximum delegation depth ({MAX_DELEGATION_DEPTH}) reached. Handle this task yourself."

        # Guard: agent exists
        if agent not in agent_directory:
            available = ", ".join(n for n in agent_directory if n != self_name)
            return f"Error: Agent '{agent}' not found. Available agents: {available}"

        # Run the delegation with timeout
        try:
            result = await asyncio.wait_for(
                run_delegate(agent, task, current_depth + 1),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return f"Error: Agent '{agent}' timed out after {timeout}s."
        except Exception as e:
            return f"Error: Delegation to '{agent}' failed: {e}"

    return Tool(
        name="delegate_task",
        description=(
            f"Delegate a sub-task to another agent and get the result back.\n\n"
            f"Available agents:\n{directory_text}"
        ),
        when_to_use=(
            "When you need help from a specialist agent to complete part of your task. "
            "For example, if you need a database schema, delegate to the database agent."
        ),
        parameters=[
            ToolParameter(
                name="agent",
                type="string",
                description="Name of the agent to delegate to",
                required=True,
            ),
            ToolParameter(
                name="task",
                type="string",
                description="Clear description of what you need the agent to do",
                required=True,
            ),
        ],
        function=delegate_task,
    )
