"""
Orchestrator for Veska.

The manager agent. Takes user input, creates a plan, assigns tasks to agents,
manages execution, handles communication, and collects final results.

Responsibilities:
  - Takes user prompt + config
  - Creates plan using AI (breaks into tasks with dependencies)
  - Assigns tasks to agents
  - Manages hybrid parallel/sequential execution
  - Watches message bus
  - Handles checkpoints (pause for user approval)
  - Manages error recovery flow
  - Collects final results
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Union

from veska.core.agent import Agent, AgentResult
from veska.core.context_manager import ContextManager
from veska.core.events import EventEmitter, EventType, Event
from veska.core.memory import SharedMemory
from veska.core.message_bus import MessageBus, BusMessage, MessageType
from veska.core.prompt_manager import PromptManager
from veska.core.task_planner import TaskPlanner, Task, TaskStatus
from veska.core.thinking import ThinkingHandler
from veska.core.helpers import resolve_provider
from veska.providers.base import BaseProvider, Message
from veska.tools.base import Tool
from veska.tools.delegation import create_delegation_tool
from veska.tools.registry import ToolRegistry


class OrchestratorConfig:
    """Configuration for the Orchestrator."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        tools: Optional[list[str | Tool]] = None,
        agents: Optional[dict[str, Agent]] = None,
        thinking: Optional[dict] = None,
        interaction_level: str = "balanced",  # minimal, balanced, detailed
        storage_dir: Optional[str] = None,
        # Clarification (off by default)
        clarification_prompt: Optional[str] = None,
        on_ask_user: Optional[Any] = None,
        # Delegation (off by default)
        allow_delegation: bool = False,
        delegation_timeout: int = 300,
        # Optional systems (off by default)
        tracking: Optional[dict] = None,
        recovery: Optional[dict] = None,
        security: Optional[dict] = None,
        mcp_servers: Optional[list[dict]] = None,
        logging: Optional[dict] = None,
    ) -> None:
        self.provider = provider
        self.tools = tools or []
        self.agents = agents or {}
        self.thinking = thinking or {}
        self.interaction_level = interaction_level
        self.storage_dir = storage_dir
        self.clarification_prompt = clarification_prompt
        self.on_ask_user = on_ask_user
        self.allow_delegation = allow_delegation
        self.delegation_timeout = delegation_timeout
        self.tracking = tracking
        self.recovery = recovery
        self.security = security
        self.mcp_servers = mcp_servers
        self.logging = logging


class Orchestrator:
    """
    The brain of Veska. Manages the entire multi-agent workflow.

    Usage:
        orch = Orchestrator(
            model="claude-sonnet-4-6",
            agents=[researcher, writer],
            tools=["file_manager"],
        )

        result = orch.run("Build me a blog app")
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[BaseProvider] = None,
        tools: Optional[list[str | Tool]] = None,
        agents: Optional[list[Agent]] = None,
        thinking: Optional[dict] = None,
        interaction_level: str = "balanced",
        storage_dir: Optional[str] = None,
        clarification_prompt: Optional[str] = None,
        on_ask_user: Optional[Any] = None,
        allow_delegation: bool = False,
        delegation_timeout: int = 300,
        max_tokens: int = 8096,
        tracking: Optional[dict] = None,
        recovery: Optional[dict] = None,
        security: Optional[dict] = None,
        mcp_servers: Optional[list[dict]] = None,
        logging: Optional[dict] = None,
    ) -> None:
        # Resolve provider: use passed provider, or create one from model/api_key
        if provider:
            self.provider = provider
        elif model:
            self.provider = resolve_provider(model=model, api_key=api_key, max_tokens=max_tokens)
        else:
            self.provider = resolve_provider(api_key=api_key, max_tokens=max_tokens)

        # Store config values for internal use
        self._clarification_prompt = clarification_prompt
        self._on_ask_user = on_ask_user
        self._interaction_level = interaction_level
        self._allow_delegation = allow_delegation
        self._delegation_timeout = delegation_timeout

        # Core systems
        self.message_bus = MessageBus()
        self.events = EventEmitter()
        self.shared_memory = SharedMemory()
        self.task_planner = TaskPlanner()
        self.context = ContextManager(
            agent_id="orchestrator",
            storage_dir=storage_dir,
        )

        # Tool registry
        self.tool_registry = ToolRegistry()
        for tool in (tools or []):
            self.tool_registry.register(tool)

        # Agents: build lookup dict from agent.name
        self._agents: dict[str, Agent] = {a.name: a for a in (agents or [])}

        # Delegation
        if allow_delegation and self._agents:
            self._setup_delegation(delegation_timeout)

        # Thinking support for orchestrator's own planning
        self._thinking = ThinkingHandler(**(thinking or {}))

        # State
        self._status: str = "idle"  # idle, planning, running, paused, done, failed
        self._paused: bool = False
        self._cancelled: bool = False
        self._current_plan: Optional[dict] = None

        # Watch all messages on the bus
        self.message_bus.add_watcher(self._on_message)

        # Listen for pause/resume/cancel events
        self.events.on(EventType.PAUSED, self._on_pause)
        self.events.on(EventType.RESUMED, self._on_resume)
        self.events.on(EventType.CANCELLED, self._on_cancel)
        self.events.on(EventType.USER_FEEDBACK, self._on_feedback)

    # --- Clarification ---

    async def _clarify(self, prompt: str) -> str:
        """
        Ask the user clarifying questions before planning.

        Uses the developer's clarification_prompt to guide the AI on what to ask.
        The AI decides if questions are needed based on the user's request.
        Returns the user's answers to append to the prompt, or empty string if skipped.
        """
        if not self._clarification_prompt or not self._on_ask_user:
            return ""

        if not self.provider:
            return ""

        # Ask AI: "Given this user request and the developer's guidance, what questions should I ask?"
        system = f"""You are the Orchestrator of an AI agent system.

The developer has provided guidance on what to clarify before starting work:

{self._clarification_prompt}

Your job:
1. Read the user's request below
2. Decide if you have enough information to create a good plan
3. If the request is already detailed enough, respond with exactly: NO_QUESTIONS_NEEDED
4. If you need more info, respond with a clear, friendly message asking the user your questions. Keep it concise — only ask what's actually unclear. Number your questions."""

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ]

        response = await self.provider.chat(messages=messages)
        ai_response = response.content.strip()

        # AI decided no questions needed
        if "NO_QUESTIONS_NEEDED" in ai_response:
            return ""

        # Ask the user via the callback
        import asyncio
        import inspect

        callback = self._on_ask_user

        try:
            if inspect.iscoroutinefunction(callback):
                user_answer = await asyncio.wait_for(
                    callback(ai_response), timeout=300
                )
            else:
                user_answer = await asyncio.wait_for(
                    asyncio.to_thread(callback, ai_response), timeout=300
                )
        except asyncio.TimeoutError:
            return ""

        if not user_answer:
            return ""

        return f"\n\nUser's clarifications:\n{user_answer}"

    # --- Delegation ---

    def _setup_delegation(self, timeout: int = 300) -> None:
        """Register delegate_task tool on all agents."""
        # Build agent directory: {name: system_prompt_summary}
        agent_directory = {}
        for name, agent in self._agents.items():
            # Use first 100 chars of system prompt as description
            desc = agent.prompt_manager.developer_prompt[:100] if agent.prompt_manager.developer_prompt else name
            agent_directory[name] = desc

        # Give each agent a delegation tool
        for name, agent in self._agents.items():
            delegation_tool = create_delegation_tool(
                agent_directory=agent_directory,
                run_delegate=self._run_delegate,
                self_name=name,
                current_depth=0,
                timeout=timeout,
            )
            agent.update_tools(agent.tools + [delegation_tool])

    async def _run_delegate(self, agent_name: str, task: str, depth: int) -> str:
        """Execute a delegated task on the target agent."""
        agent = self._agents.get(agent_name)
        if not agent:
            return f"Error: Agent '{agent_name}' not found."

        # If delegation is chained, update the delegation tool depth on the target
        # so it knows its current depth for guard rails
        if depth > 0:
            for i, tool in enumerate(agent.tools):
                if tool.name == "delegate_task":
                    # Rebuild with incremented depth
                    agent_directory = {}
                    for n, a in self._agents.items():
                        desc = a.prompt_manager.developer_prompt[:100] if a.prompt_manager.developer_prompt else n
                        agent_directory[n] = desc

                    agent.tools[i] = create_delegation_tool(
                        agent_directory=agent_directory,
                        run_delegate=self._run_delegate,
                        self_name=agent_name,
                        current_depth=depth,
                        timeout=self._delegation_timeout,
                    )
                    agent._tool_map[agent.tools[i].name] = agent.tools[i]
                    break

        result = await agent.run(task=task)
        return result.output if result.success else f"Error: {result.error}"

    # --- Agent management ---

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.name] = agent

        # Subscribe agent to message bus
        async def agent_message_handler(msg: BusMessage) -> None:
            # Store incoming messages in agent's memory
            agent.memory.add(
                key=f"msg_from_{msg.from_agent}",
                value=msg.content[:200],
                category="messages",
            )

        self.message_bus.subscribe(agent.name, agent_message_handler)

        # Store agent's memory in shared memory
        self.shared_memory.store(agent.memory)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self._agents.get(name)

    # --- Main execution ---

    async def arun(self, prompt: str) -> OrchestratorResult:
        """
        Async entry point. Use this inside FastAPI, Jupyter, or any async app.

        Args:
            prompt: What you want the agents to do.
        """
        return await self._run_async(prompt)

    def run(self, prompt: str) -> OrchestratorResult:
        """
        Sync entry point for scripts. Do NOT call inside a running event loop.
        Use: result = await orchestrator.arun(...) instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "orchestrator.run() was called inside a running event loop. "
                "Use: result = await orchestrator.arun(...) instead."
            )
        except RuntimeError as e:
            if "running event loop" in str(e):
                raise
            pass

        return asyncio.run(self.arun(prompt))

    async def _run_async(self, prompt: str) -> OrchestratorResult:
        """Internal async implementation."""
        self._status = "planning"
        self._cancelled = False
        self._paused = False

        await self.events.emit(Event(
            type=EventType.STARTED,
            source="orchestrator",
            message=f"Starting: {prompt[:100]}",
        ))

        try:
            # Step 0: Clarify (if developer set clarification_prompt)
            clarifications = await self._clarify(prompt)
            if clarifications:
                prompt = prompt + clarifications

            # Step 1: Create the plan
            plan = await self._create_plan(prompt)
            if not plan:
                return OrchestratorResult(
                    success=False, error="Failed to create plan"
                )

            self._current_plan = plan

            # Step 2: Checkpoint - show plan to user
            if self._interaction_level != "minimal":
                checkpoint_response = await self.events.checkpoint(
                    checkpoint_id="plan_review",
                    title="Review Plan",
                    description="Here's the plan. Approve or suggest changes.",
                    details=plan,
                )

                if not checkpoint_response.get("approved"):
                    feedback = checkpoint_response.get("feedback", "")
                    if feedback:
                        # Re-plan with user feedback
                        plan = await self._create_plan(
                            f"{prompt}\n\nUser feedback: {feedback}"
                        )
                        self._current_plan = plan
                    else:
                        return OrchestratorResult(
                            success=False, error="Plan rejected by user"
                        )

            # Step 3: Build tasks from plan
            self._build_tasks_from_plan(plan)

            # Step 4: Execute tasks
            self._status = "running"
            await self._execute_tasks()

            # Step 5: Collect results
            results = self._collect_results()

            self._status = "done"
            await self.events.emit(Event(
                type=EventType.COMPLETED,
                source="orchestrator",
                message="All tasks completed",
                data=results,
            ))

            return OrchestratorResult(
                success=not self.task_planner.has_failures,
                plan=plan,
                results=results,
                progress=self.task_planner.progress,
            )

        except Exception as e:
            self._status = "failed"
            await self.events.emit_error("orchestrator", str(e))
            return OrchestratorResult(success=False, error=str(e))

    async def _create_plan(self, prompt: str) -> Optional[dict]:
        """Use AI to break the user's prompt into a structured plan."""
        if not self.provider:
            return self._create_default_plan(prompt)

        planning_prompt = self._build_planning_prompt(prompt)

        messages = [
            Message(role="system", content=planning_prompt),
            Message(role="user", content=prompt),
        ]

        response = await self.provider.chat(
            messages=messages,
            thinking=self._thinking.get_config() if self._thinking.enabled else None,
        )

        # Handle thinking output
        if response.thinking:
            self._thinking.process(response.thinking, task_id="planning")

        # Parse plan from response
        return self._parse_plan(response.content)

    def _build_planning_prompt(self, prompt: str) -> str:
        """Build the system prompt for planning."""
        agent_list = ", ".join(self._agents.keys()) if self._agents else "no agents registered yet"
        tool_list = ", ".join(self.tool_registry.get_names()) if len(self.tool_registry) > 0 else "none"

        return f"""You are the Orchestrator of a multi-agent system called Veska.

Your job is to take a user's request and break it into a structured plan.

Available agents: {agent_list}
Available tools: {tool_list}

Create a plan as a JSON object with this structure:
{{
    "name": "short plan name",
    "description": "what this plan will achieve",
    "phases": [
        {{
            "name": "phase name",
            "description": "what this phase does",
            "tasks": [
                {{
                    "id": "unique_task_id",
                    "name": "task name",
                    "description": "detailed description of what to do",
                    "agent": "agent_name",
                    "depends_on": ["task_id_1", "task_id_2"]
                }}
            ]
        }}
    ]
}}

Rules:
- Break work into small, focused tasks
- Set dependencies correctly (task B depends on task A if B needs A's output)
- Tasks in the same phase with no dependencies between them can run in parallel
- Assign each task to the most appropriate agent
- Be specific in task descriptions so agents know exactly what to do

Respond with ONLY the JSON plan, no other text."""

    def _parse_plan(self, content: str) -> Optional[dict]:
        """Parse the AI's response into a plan dict."""
        try:
            # Try to extract JSON from the response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.startswith("```") and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            return json.loads(content)
        except json.JSONDecodeError:
            # If AI didn't return valid JSON, try to find JSON in the text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    return None
            return None

    def _create_default_plan(self, prompt: str) -> dict:
        """Create a basic plan when no AI provider is available."""
        return {
            "name": "Default Plan",
            "description": prompt,
            "phases": [
                {
                    "name": "Execution",
                    "description": "Execute the request",
                    "tasks": [
                        {
                            "id": "main_task",
                            "name": "Execute request",
                            "description": prompt,
                            "agent": list(self._agents.keys())[0] if self._agents else "default",
                            "depends_on": [],
                        }
                    ],
                }
            ],
        }

    def _build_tasks_from_plan(self, plan: dict) -> None:
        """Convert the plan dict into Task objects in the TaskPlanner."""
        self.task_planner = TaskPlanner()

        for phase in plan.get("phases", []):
            task_ids = []
            for task_data in phase.get("tasks", []):
                task = Task(
                    id=task_data["id"],
                    name=task_data["name"],
                    description=task_data.get("description", ""),
                    agent=task_data["agent"],
                    depends_on=task_data.get("depends_on", []),
                )
                self.task_planner.add_task(task)
                task_ids.append(task.id)

            self.task_planner.add_phase(
                name=phase["name"],
                task_ids=task_ids,
                description=phase.get("description", ""),
            )

        # Validate the plan
        errors = self.task_planner.validate()
        if errors:
            raise ValueError(f"Invalid plan: {'; '.join(errors)}")

    # --- Task execution ---

    async def _execute_tasks(self) -> None:
        """Execute all tasks in dependency order using hybrid parallel/sequential."""
        while not self.task_planner.is_complete and not self._cancelled:
            # Check if paused
            if self._paused:
                await asyncio.sleep(0.5)
                continue

            # Get tasks that are ready to run
            ready_tasks = self.task_planner.get_ready_tasks()

            if not ready_tasks:
                if self.task_planner.has_failures:
                    # All remaining tasks are blocked by failures
                    break
                # No ready tasks but not complete - something's wrong
                await asyncio.sleep(0.1)
                continue

            # Run ready tasks in parallel
            coroutines = [self._run_task(task) for task in ready_tasks]
            await asyncio.gather(*coroutines, return_exceptions=True)

            # Emit progress
            progress = self.task_planner.progress
            await self.events.emit_progress(
                source="orchestrator",
                completed=progress["completed"],
                total=progress["total"],
                current_task=ready_tasks[0].name if ready_tasks else "",
            )

            # Check for phase completion (checkpoint opportunity)
            current_phase = self.task_planner.get_current_phase()
            if current_phase and self._interaction_level == "detailed":
                prev_phase = self._get_previous_phase(current_phase.name)
                if prev_phase and self.task_planner.is_phase_complete(prev_phase):
                    await self.events.checkpoint(
                        checkpoint_id=f"phase_{prev_phase}",
                        title=f"Phase Complete: {prev_phase}",
                        description=f"Phase '{prev_phase}' is done. Continue?",
                    )

    async def _run_task(self, task: Task) -> None:
        """Run a single task by assigning it to the appropriate agent."""
        agent = self._agents.get(task.agent)
        if not agent:
            self.task_planner.fail_task(
                task.id, f"Agent '{task.agent}' not found"
            )
            await self.events.emit_error(
                "orchestrator", f"Agent '{task.agent}' not found for task '{task.name}'"
            )
            return

        # Start the task
        self.task_planner.start_task(task.id)

        await self.events.emit(Event(
            type=EventType.TASK_STARTED,
            source=task.agent,
            message=f"Starting: {task.name}",
            data={"task_id": task.id, "task_name": task.name},
        ))

        # Build context from completed dependencies
        context = self._build_task_context(task)

        # Run the agent
        try:
            result = await agent.run(
                task=task.description or task.name,
                context=context,
            )

            if result.success:
                self.task_planner.complete_task(task.id, result.output)

                # Store in context manager
                self.context.complete_task(
                    task_id=task.id,
                    summary=result.output[:200],
                    key_facts=[],
                    files_created=[],
                    full_output=result.output,
                )

                # Update shared memory
                self.shared_memory.store(agent.memory)

                await self.events.emit(Event(
                    type=EventType.TASK_COMPLETED,
                    source=task.agent,
                    message=f"Completed: {task.name}",
                    data={"task_id": task.id, "result": result.output[:200]},
                ))
            else:
                self.task_planner.fail_task(task.id, result.error or "Unknown error")

                await self.events.emit(Event(
                    type=EventType.TASK_FAILED,
                    source=task.agent,
                    message=f"Failed: {task.name} - {result.error}",
                    data={"task_id": task.id, "error": result.error},
                ))

                # Auto-retry if possible
                if self.task_planner.get_task(task.id).can_retry:
                    self.task_planner.retry_task(task.id)

        except Exception as e:
            self.task_planner.fail_task(task.id, str(e))
            await self.events.emit_error(task.agent, f"Task '{task.name}' crashed: {e}")

    def _build_task_context(self, task: Task) -> str:
        """Build context for a task from its completed dependencies."""
        parts = []

        for dep_id in task.depends_on:
            dep_task = self.task_planner.get_task(dep_id)
            if dep_task and dep_task.result:
                parts.append(
                    f"Result from '{dep_task.name}': {dep_task.result[:500]}"
                )

        # Add relevant agent memories
        for dep_id in task.depends_on:
            dep_task = self.task_planner.get_task(dep_id)
            if dep_task:
                dep_summary = self.shared_memory.get_summary(dep_task.agent)
                if dep_summary:
                    parts.append(f"\n{dep_task.agent}'s summary:\n{dep_summary}")

        return "\n\n".join(parts)

    def _get_previous_phase(self, current_phase_name: str) -> Optional[str]:
        """Get the name of the phase before the current one."""
        phases = [p.name for p in self.task_planner._phases]
        idx = phases.index(current_phase_name) if current_phase_name in phases else -1
        if idx > 0:
            return phases[idx - 1]
        return None

    # --- Results ---

    def _collect_results(self) -> dict:
        """Collect results from all completed tasks."""
        results = {
            "tasks": {},
            "agents": {},
        }

        for task in self.task_planner.get_all_tasks():
            results["tasks"][task.id] = {
                "name": task.name,
                "agent": task.agent,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
                "duration": task.duration,
            }

        for name, agent in self._agents.items():
            results["agents"][name] = {
                "status": agent.status,
                "memory_summary": agent.memory.get_summary(),
                "tasks_done": len(agent.memory.get_tasks()),
            }

        return results

    # --- Event handlers ---

    async def _on_message(self, message: BusMessage) -> None:
        """Watch all messages on the bus."""
        # Orchestrator sees everything but only acts on specific types
        if message.type == MessageType.ERROR:
            await self.events.emit_error(
                message.from_agent, message.content
            )

    async def _on_pause(self, event: Event) -> None:
        """Handle pause event."""
        self._paused = True
        self._status = "paused"

    async def _on_resume(self, event: Event) -> None:
        """Handle resume event."""
        self._paused = False
        self._status = "running"

    async def _on_cancel(self, event: Event) -> None:
        """Handle cancel event."""
        self._cancelled = True
        self._status = "failed"

        # Cancel all running tasks
        for task in self.task_planner.get_tasks_by_status(TaskStatus.RUNNING):
            self.task_planner.cancel_task(task.id)

    async def _on_feedback(self, event: Event) -> None:
        """Handle user feedback."""
        # Store feedback for potential re-planning
        self.context.complete_task(
            task_id="user_feedback",
            summary=event.message,
            key_facts=[event.message],
        )

    # --- Properties ---

    @property
    def status(self) -> str:
        return self._status

    @property
    def progress(self) -> dict:
        return self.task_planner.progress


class OrchestratorResult:
    """Result from an orchestrator run."""

    def __init__(
        self,
        success: bool,
        plan: Optional[dict] = None,
        results: Optional[dict] = None,
        progress: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.plan = plan
        self.results = results
        self.progress = progress
        self.error = error

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"OrchestratorResult({status})"
