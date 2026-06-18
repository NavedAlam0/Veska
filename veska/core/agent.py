"""
Base Agent class for Veska.

The core building block. Every agent follows:
  Think -> Decide -> Act -> Observe -> Repeat

An agent has:
  - A role (system prompt from developer)
  - A model (Claude/OpenAI, configurable)
  - Tools (unified list - pre-built + custom + MCP)
  - Memory (private, shareable)
  - Context management (stays within token limits)
  - Thinking support (optional, per agent)

General purpose - works for any use case.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from veska.core.context_manager import ContextManager
from veska.core.memory import AgentMemory
from veska.core.prompt_manager import PromptManager
from veska.core.thinking import ThinkingHandler
from veska.providers.base import BaseProvider, Message, ProviderResponse
from veska.tools.base import Tool, ToolResult


class AgentConfig:
    """Configuration for creating an agent."""

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        provider: Optional[BaseProvider] = None,
        tools: Optional[list[Tool]] = None,
        thinking: Optional[dict] = None,
        max_iterations: int = 20,
        storage_dir: Optional[str] = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider
        self.tools = tools or []
        self.thinking = thinking or {}
        self.max_iterations = max_iterations
        self.storage_dir = storage_dir


class Agent:
    """
    Base agent class for Veska framework.

    Usage:
        agent = Agent(AgentConfig(
            name="backend_developer",
            system_prompt="You are a senior Python developer...",
            provider=ClaudeProvider(model="claude-sonnet-4-6"),
            tools=[file_manager, code_runner],
            thinking={"enabled": True, "budget_tokens": 10000, "output": "log"},
        ))

        # Run a task
        result = await agent.run("Create user authentication API")

        # Get agent's memory
        summary = agent.memory.get_summary()
    """

    def __init__(self, config: AgentConfig) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.name = config.name
        self.provider = config.provider
        self.max_iterations = config.max_iterations

        # Tools
        self.tools = config.tools
        self._tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # Memory (private)
        self.memory = AgentMemory(agent_id=self.name)

        # Context manager
        self.context = ContextManager(
            agent_id=self.name,
            storage_dir=config.storage_dir,
        )

        # Thinking handler
        self.thinking = ThinkingHandler(**config.thinking)

        # Prompt manager
        self.prompt_manager = PromptManager(
            developer_prompt=config.system_prompt,
            tools=self.tools,
        )

        # Conversation history
        self._messages: list[Message] = []

        # State
        self._status: str = "idle"  # idle, working, waiting, done, failed

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value
        self.memory.set_state(value)

    async def run(self, task: str, context: str = "") -> AgentResult:
        """
        Run a task. This is the main Think -> Decide -> Act -> Observe loop.

        Args:
            task: The task description.
            context: Additional context (summaries from other tasks, etc.)

        Returns:
            AgentResult with the final output.
        """
        if not self.provider:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output="",
                error="No provider configured",
            )

        self.status = "working"

        # Build system prompt with task context
        task_context = task
        if context:
            task_context = f"{context}\n\n{task}"
        self.prompt_manager.update_task_context(task_context)
        system_prompt = self.prompt_manager.build()

        # Start conversation
        self._messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]

        # Get provider format for tools
        tool_defs = (
            [t.to_provider_format(self.provider.provider_name) for t in self.tools]
            if self.tools
            else None
        )

        # Think -> Decide -> Act -> Observe loop
        iteration = 0
        final_output = ""

        while iteration < self.max_iterations:
            iteration += 1

            # Think + Decide: Ask the AI model
            response = await self.provider.chat(
                messages=self._messages,
                tools=tool_defs,
                thinking=self.thinking.get_config() if self.thinking.enabled else None,
            )

            # Handle thinking output
            if response.thinking:
                self.thinking.process(response.thinking, task_id=task)

            # Act: If model wants to use tools
            if response.has_tool_calls:
                # Record assistant message with tool calls
                self._messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_result = await self._execute_tool(
                        tool_call["name"],
                        tool_call["arguments"],
                    )

                    # Observe: Add tool result to conversation
                    self._messages.append(Message(
                        role="tool",
                        content=self._format_tool_result(tool_result),
                        tool_call_id=tool_call["id"],
                    ))

                # Continue the loop - let model see tool results
                continue

            # No tool calls - model is done
            final_output = response.content
            break

        # Record in memory
        self.memory.add_task(task, final_output[:200])
        self.status = "done"

        # Trim messages to prevent context overflow in future runs
        self._messages = self.context.trim_messages(self._messages)

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=final_output,
            iterations=iteration,
        )

    async def _execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self._tool_map.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        result = await tool.execute(**arguments)

        # Record in memory
        if result.success:
            self.memory.add(
                key=tool_name,
                value=str(result.output)[:100],
                category="tool_usage",
            )
        else:
            self.memory.add_error(f"Tool {tool_name} failed: {result.error}")

        return result

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format tool result for the conversation."""
        if result.success:
            output = result.output
            if isinstance(output, (dict, list)):
                return json.dumps(output, indent=2)
            return str(output)
        return f"Error: {result.error}"

    def update_tools(self, tools: list[Tool]) -> None:
        """Update the agent's available tools."""
        self.tools = tools
        self._tool_map = {t.name: t for t in tools}
        self.prompt_manager.update_tools(tools)

    def get_conversation_history(self) -> list[Message]:
        """Get the current conversation history."""
        return list(self._messages)

    def reset(self) -> None:
        """Reset the agent for a new task (keeps memory)."""
        self._messages.clear()
        self.status = "idle"


class AgentResult:
    """Result from an agent's task execution."""

    def __init__(
        self,
        agent_name: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
        iterations: int = 0,
    ) -> None:
        self.agent_name = agent_name
        self.success = success
        self.output = output
        self.error = error
        self.iterations = iterations

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"AgentResult({self.agent_name}: {status}, iterations={self.iterations})"
