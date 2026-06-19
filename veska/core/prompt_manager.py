"""
Prompt Manager - builds the final system prompt.

Two layers:
  Layer 1 (Framework): Generic infrastructure rules. Auto-injected. NOT editable.
  Layer 2 (Developer): Custom purpose/role. Fully editable.

Framework layer is general purpose - works for any use case.
No hardcoded references to websites, code, frontend, or backend.
"""

from __future__ import annotations

from typing import Optional

from veska.tools.base import Tool

# Framework layer - generic infrastructure rules
# Works for ANY use case: code generation, customer support, data analysis, etc.
FRAMEWORK_PROMPT = """You are an agent in a multi-agent system.

## Communication
- You receive tasks from the Orchestrator.
- You send results back when your task is complete.
- You can request information from other agents through the message bus.
- You report errors immediately with clear descriptions.

## Tools
- You have specific tools available to complete your tasks.
- Use tools when needed. Choose the right tool for each action.
- Follow each tool's "when to use" guidelines.
- Report tool errors clearly if they occur.

## Memory
- Remember your decisions and actions.
- Reference your memory when continuing or revisiting work.
- Keep track of what you've created, modified, or completed.

## Rules
- Answer exactly what was asked. Do not add extra information, breakdowns, or explanations unless specifically requested.
- Be concise. Give direct answers.
- Stay within your assigned scope. Do not work outside your territory.
- Follow the Orchestrator's plan and task assignments.
- If you're unsure about something, ask for clarification rather than guessing.
"""


class PromptManager:
    """
    Builds the final system prompt from all layers.

    Usage:
        pm = PromptManager(
            developer_prompt="You are a senior Python developer...",
            tools=[tool1, tool2],
        )
        system_prompt = pm.build()
    """

    def __init__(
        self,
        developer_prompt: str = "",
        tools: Optional[list[Tool]] = None,
        task_context: Optional[str] = None,
    ) -> None:
        self.developer_prompt = developer_prompt
        self.tools = tools or []
        self.task_context = task_context

    def build(self) -> str:
        """Build the complete system prompt from all layers."""
        parts = []

        # Layer 1: Framework rules (always present, not editable)
        parts.append(FRAMEWORK_PROMPT)

        # Tool documentation (auto-generated from registered tools)
        if self.tools:
            parts.append(self._build_tool_docs())

        # Layer 2: Developer prompt (custom purpose/role)
        if self.developer_prompt:
            parts.append(f"## Your Role\n{self.developer_prompt}")

        # Layer 3: Task context (injected per task by orchestrator)
        if self.task_context:
            parts.append(f"## Current Task\n{self.task_context}")

        return "\n\n".join(parts)

    def update_task_context(self, context: str) -> None:
        """Update the task context layer (called by orchestrator per task)."""
        self.task_context = context

    def update_tools(self, tools: list[Tool]) -> None:
        """Update available tools."""
        self.tools = tools

    def _build_tool_docs(self) -> str:
        """Auto-generate tool documentation for the system prompt."""
        lines = ["## Available Tools"]

        for tool in self.tools:
            lines.append(f"\n### {tool.name}")
            lines.append(f"**Description:** {tool.description}")

            if tool.when_to_use:
                lines.append(f"**When to use:** {tool.when_to_use}")

            if tool.parameters:
                lines.append("**Parameters:**")
                for param in tool.parameters:
                    req = "(required)" if param.required else "(optional)"
                    lines.append(
                        f"  - `{param.name}` ({param.type}) {req}: {param.description}"
                    )

        return "\n".join(lines)
