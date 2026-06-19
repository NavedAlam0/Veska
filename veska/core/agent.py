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
from typing import Any, AsyncGenerator, Optional, Type, Union

from pydantic import BaseModel

from veska.core.context_manager import ContextManager
from veska.core.helpers import resolve_provider
from veska.core.memory import AgentMemory
from veska.core.prompt_manager import PromptManager
from veska.core.format import strip_markdown
from veska.core.structured import build_retry_message, build_schema_instructions, dict_to_model, extract_and_validate
from veska.core.thinking import ThinkingHandler
from veska.cache.store import CacheStore
from veska.media.processor import process_attachments
from veska.memory.store import MemoryStore
from veska.sessions.store import SessionStore
from veska.providers.base import BaseProvider, Message, ProviderResponse, StreamEvent
from veska.tools.base import Tool, ToolResult
from veska.tools.human import create_ask_user_tool


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
        on_ask_user: Optional[Any] = None,
        ask_user_timeout: int = 300,
        memory_store: Optional[MemoryStore] = None,
        cache: Optional[CacheStore] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider
        self.tools = tools or []
        self.thinking = thinking or {}
        self.max_iterations = max_iterations
        self.storage_dir = storage_dir
        self.on_ask_user = on_ask_user
        self.ask_user_timeout = ask_user_timeout
        self.memory_store = memory_store
        self.cache = cache
        self.session_store = session_store


class Agent:
    """
    Base agent class for Veska framework.

    Usage:
        agent = Agent(
            name="backend_developer",
            system_prompt="You are a senior Python developer...",
            provider=ClaudeProvider(api_key="your-key", model="claude-sonnet-4-6"),
            tools=[file_manager, code_runner],
        )

        # Bulk mode (default)
        result = await agent.run("Create user authentication API")

        # Streaming mode
        async for event in agent.run("Create user authentication API", stream=True):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
    """

    def __init__(
        self,
        name: str = "",
        *,
        system_prompt: str = "",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[BaseProvider] = None,
        tools: Optional[list] = None,
        thinking: Optional[dict] = None,
        max_iterations: int = 20,
        max_tokens: int = 8096,
        temperature: Optional[float] = None,
        storage_dir: Optional[str] = None,
        on_ask_user: Optional[Any] = None,
        ask_user_timeout: int = 300,
        memory_store: Optional[MemoryStore] = None,
        cache: Optional[CacheStore] = None,
        session_store: Optional[SessionStore] = None,
        output_format: Optional[dict] = None,
    ) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.name = name

        # Structured output: convert dict format to Pydantic model internally
        self._output_model = dict_to_model(output_format) if output_format else None

        # Resolve provider: use passed provider, or create one from model/api_key
        if provider:
            self.provider = provider
        elif model:
            self.provider = resolve_provider(model=model, api_key=api_key, max_tokens=max_tokens, temperature=temperature)
        else:
            self.provider = resolve_provider(api_key=api_key, max_tokens=max_tokens, temperature=temperature)
        self.max_iterations = max_iterations

        # Tools: accept Tool objects or @tool-decorated functions
        self.tools = [
            t._tool if hasattr(t, "_tool") else t
            for t in (tools or [])
        ]

        # HITL: auto-register ask_user tool if callback provided
        if on_ask_user:
            self.tools.append(
                create_ask_user_tool(on_ask_user, ask_user_timeout)
            )

        self._tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # Memory (private, with optional persistent store)
        self.memory = AgentMemory(agent_id=self.name)
        self.memory_store = memory_store

        # Cache (optional, developer decides what to cache)
        self.cache = cache

        # Sessions (optional, persists conversation threads)
        self.session_store = session_store

        # Context manager
        self.context = ContextManager(
            agent_id=self.name,
            storage_dir=storage_dir,
        )

        # Thinking handler
        self.thinking = ThinkingHandler(**(thinking or {}))

        # Prompt manager
        self.prompt_manager = PromptManager(
            developer_prompt=system_prompt,
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

    async def arun(
        self,
        task: str,
        context: str = "",
        stream: Union[bool, callable, None] = None,
        output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Async entry point. Use this inside FastAPI, Jupyter, or any async app.

        Args:
            task: The task description.
            context: Additional context.
            stream: None/False = no streaming, returns result.
                    True = stream tokens to console (print).
                    callable = stream tokens to your function.
            output_model: Optional Pydantic model for structured output.
            attachments: Optional list of file paths, URLs, or Image/PDF objects.
            user_id: Optional user identifier for session persistence.
            session_id: Optional session identifier for conversation resumption.
        """
        effective_model = output_model or self._output_model

        if stream:
            if stream is True:
                callback = lambda text: print(text, end="", flush=True)
            else:
                callback = stream

            return await self._run_with_stream(callback, task, context, effective_model, attachments, user_id, session_id)

        return await self._run_bulk(task, context, effective_model, attachments, user_id, session_id)

    def run(
        self,
        task: str,
        context: str = "",
        stream: Union[bool, callable, None] = None,
        output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """
        Sync entry point for scripts. Do NOT call inside a running event loop.
        Use await agent.arun(...) instead for async apps.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "agent.run() was called inside a running event loop. "
                "Use: result = await agent.arun(...) instead."
            )
        except RuntimeError as e:
            if "running event loop" in str(e):
                raise
            pass

        return asyncio.run(
            self.arun(task, context, stream, output_model, attachments, user_id, session_id)
        )

    async def _run_with_stream(
        self,
        callback: callable,
        task: str,
        context: str = "",
        output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Run with streaming, calling the callback for each text token."""
        result = None
        async for event in self._run_stream(task, context, output_model, attachments, user_id, session_id):
            if event.type == "text_delta":
                callback(strip_markdown(event.text))
            elif event.type == "done":
                result = AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=strip_markdown(event.response.content) if event.response else "",
                    iterations=0,
                )
                if event.parsed_output:
                    result.output = event.parsed_output.model_dump() if self._output_model and not output_model else event.parsed_output
        return result

    async def _run_bulk(
        self, task: str, context: str = "", output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None, user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """Run a task and return the full result at once."""
        if not self.provider:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output="",
                error="No provider configured",
            )

        self.status = "working"

        # Load relevant past memories if persistent store exists
        memory_context = await self._load_memories(task)
        full_context = f"{memory_context}\n\n{context}".strip() if memory_context else context

        self._setup_conversation(task, full_context, output_model, attachments)

        # Session: load previous conversation and prepend
        await self._load_session(user_id, session_id)
        tool_defs = self._get_tool_defs()

        iteration = 0
        final_output = ""
        parsed_output = None
        validation_retries = 0
        max_validation_retries = 3

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=self._messages,
                tools=tool_defs,
                thinking=self.thinking.get_config() if self.thinking.enabled else None,
            )

            if response.thinking:
                self.thinking.process(response.thinking, task_id=task)

            if response.has_tool_calls:
                self._messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    tool_result = await self._execute_tool(
                        tool_call["name"],
                        tool_call["arguments"],
                    )
                    self._messages.append(Message(
                        role="tool",
                        content=self._format_tool_result(tool_result),
                        tool_call_id=tool_call["id"],
                    ))

                continue

            final_output = response.content

            # Structured output: validate and retry if needed
            if output_model:
                parsed, error = extract_and_validate(output_model, final_output)
                if parsed:
                    parsed_output = parsed
                    self._messages.append(Message(role="assistant", content=final_output))
                    break

                validation_retries += 1
                if validation_retries >= max_validation_retries:
                    self.status = "done"
                    return AgentResult(
                        agent_name=self.name,
                        success=False,
                        output=final_output,
                        error=f"Failed to produce valid structured output after {max_validation_retries} attempts. Last error: {error}",
                    )

                # Send error back to AI for retry
                self._messages.append(Message(role="assistant", content=final_output))
                self._messages.append(Message(role="user", content=build_retry_message(error)))
                continue

            self._messages.append(Message(role="assistant", content=final_output))
            break

        self.memory.add_task(task, final_output[:200])
        await self._save_memory(task, final_output[:200])
        await self._save_session(user_id, session_id)
        self.status = "done"
        self._messages = self.context.trim_messages(self._messages)

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=parsed_output.model_dump() if parsed_output and self._output_model and not output_model else (parsed_output if parsed_output else strip_markdown(final_output)),
            iterations=iteration,
        )

    async def _run_stream(
        self, task: str, context: str = "", output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None, user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run a task, yielding StreamEvent objects as tokens arrive."""
        if not self.provider:
            yield StreamEvent(
                type="done",
                response=ProviderResponse(content="No provider configured"),
            )
            return

        self.status = "working"

        # Load relevant past memories if persistent store exists
        memory_context = await self._load_memories(task)
        full_context = f"{memory_context}\n\n{context}".strip() if memory_context else context

        self._setup_conversation(task, full_context, output_model, attachments)

        # Session: load previous conversation and prepend
        await self._load_session(user_id, session_id)

        tool_defs = self._get_tool_defs()

        iteration = 0
        final_output = ""
        parsed_output = None
        validation_retries = 0
        max_validation_retries = 3

        while iteration < self.max_iterations:
            iteration += 1

            full_response = None

            async for event in await self.provider.chat(
                messages=self._messages,
                tools=tool_defs,
                thinking=self.thinking.get_config() if self.thinking.enabled else None,
                stream=True,
            ):
                if event.type == "done":
                    full_response = event.response
                else:
                    yield event

            if full_response is None:
                break

            if full_response.thinking:
                self.thinking.process(full_response.thinking, task_id=task)

            if full_response.has_tool_calls:
                self._messages.append(Message(
                    role="assistant",
                    content=full_response.content,
                    tool_calls=full_response.tool_calls,
                ))

                for tool_call in full_response.tool_calls:
                    tool_result = await self._execute_tool(
                        tool_call["name"],
                        tool_call["arguments"],
                    )
                    result_text = self._format_tool_result(tool_result)
                    self._messages.append(Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=tool_call["id"],
                    ))
                    yield StreamEvent(
                        type="tool_result",
                        tool_name=tool_call["name"],
                        tool_result=result_text,
                    )

                continue

            final_output = full_response.content

            # Structured output: validate and retry if needed
            if output_model:
                parsed, error = extract_and_validate(output_model, final_output)
                if parsed:
                    parsed_output = parsed
                    self._messages.append(Message(role="assistant", content=final_output))
                    break

                validation_retries += 1
                if validation_retries >= max_validation_retries:
                    break

                self._messages.append(Message(role="assistant", content=final_output))
                self._messages.append(Message(role="user", content=build_retry_message(error)))
                continue

            self._messages.append(Message(role="assistant", content=final_output))
            break

        self.memory.add_task(task, final_output[:200])
        await self._save_memory(task, final_output[:200])
        await self._save_session(user_id, session_id)
        self.status = "done"
        self._messages = self.context.trim_messages(self._messages)

        yield StreamEvent(
            type="done",
            response=ProviderResponse(
                content=final_output,
                model=self.provider.model,
            ),
            parsed_output=parsed_output,
        )

    def _setup_conversation(
        self, task: str, context: str, output_model: Optional[Type[BaseModel]] = None,
        attachments: Optional[list] = None,
    ) -> None:
        """Prepare system prompt and initial messages."""
        task_context = task
        if context:
            task_context = f"{context}\n\n{task}"
        self.prompt_manager.update_task_context(task_context)
        system_prompt = self.prompt_manager.build()

        if output_model:
            system_prompt += build_schema_instructions(output_model)

        # Build user message — text-only or multi-modal
        if attachments:
            content_blocks = process_attachments(attachments)
            # Prepend the task text
            user_content: list[dict] = [{"type": "text", "text": task}] + content_blocks
            user_message = Message(role="user", content=user_content)
        else:
            user_message = Message(role="user", content=task)

        self._messages = [
            Message(role="system", content=system_prompt),
            user_message,
        ]

    def _get_tool_defs(self) -> Optional[list[dict]]:
        """Get tool definitions in provider format."""
        if not self.tools or not self.provider:
            return None
        return [t.to_provider_format(self.provider.provider_name) for t in self.tools]

    async def _load_session(self, user_id: Optional[str], session_id: Optional[str]) -> None:
        """Load previous conversation from session store and prepend to messages."""
        if not self.session_store or not user_id or not session_id:
            return
        past_messages = await self.session_store.load(user_id, session_id)
        if past_messages:
            # Insert past messages after system prompt, before current user message
            system = self._messages[0]  # system prompt
            current = self._messages[1:]  # current user message
            self._messages = [system] + past_messages + current

    async def _save_session(self, user_id: Optional[str], session_id: Optional[str]) -> None:
        """Save conversation to session store (excludes system prompt)."""
        if not self.session_store or not user_id or not session_id:
            return
        # Save everything except the system prompt
        conversation = [m for m in self._messages if m.role != "system"]
        await self.session_store.save(user_id, session_id, conversation)

    async def _load_memories(self, task: str) -> str:
        """Load relevant past memories from persistent store."""
        if not self.memory_store:
            return ""
        past = await self.memory_store.search(task, limit=10)
        if not past:
            return ""
        lines = ["Here's what you remember from past tasks:"]
        for mem in past:
            lines.append(f"- {mem.key}: {mem.value}")
        return "\n".join(lines)

    async def _save_memory(self, task: str, output: str) -> None:
        """Save task result to persistent store."""
        if not self.memory_store:
            return
        await self.memory_store.save(
            key=task[:200],
            value=output,
            metadata={"agent": self.name, "category": "task"},
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
        output: Any,
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
