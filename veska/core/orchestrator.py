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
from pathlib import Path
from typing import Any, Optional, Union

from veska.core.agent import Agent, AgentResult
from veska.core.context_manager import ContextManager
from veska.core.events import EventEmitter, EventType, Event
from veska.core.memory import AgentMemory, SharedMemory
from veska.core.message_bus import MessageBus, BusMessage, MessageType
from veska.core.prompt_manager import PromptManager
from veska.core.task_planner import TaskPlanner, Task, TaskStatus
from veska.core.thinking import ThinkingHandler
from veska.core.helpers import resolve_provider
from veska.providers.base import BaseProvider, Message
from veska.tools.base import Tool
from veska.tools.delegation import create_delegation_tool
from veska.tools.registry import ToolRegistry
from veska.logging.logger import Logger
from veska.tracking.cost_tracker import CostTracker
from veska.core.mcp_connector import MCPConnector, MCPServer
from veska.recovery.recovery import RecoveryManager, SavePoint
from veska.security.command_guard import CommandGuard
from veska.security.sandbox import Sandbox
from veska.tools.code_runner import get_code_runner_tools
from veska.tools.file_manager import get_file_manager_tools


class OrchestratorConfig:
    """Configuration for the Orchestrator."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        tools: Optional[list[str | Tool]] = None,
        agents: Optional[dict[str, Agent]] = None,
        thinking: Optional[dict] = None,
        interaction_level: str = "minimal",  # minimal, balanced, detailed
        storage_dir: Optional[str] = None,
        # Clarification (off by default)
        clarification_prompt: Optional[str] = None,
        on_ask_user: Optional[Any] = None,
        # Delegation (off by default)
        allow_delegation: bool = False,
        delegation_timeout: int = 300,
        share_tools_with_agents: bool = False,
        # Optional systems (off by default)
        recovery: Optional[dict | RecoveryManager] = None,
        security: Optional[dict | Sandbox] = None,
        mcp_servers: Optional[list[dict | MCPServer]] = None,
        logger: Optional[Logger] = None,
        cost_tracker: Optional[CostTracker] = None,
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
        self.share_tools_with_agents = share_tools_with_agents
        self.recovery = recovery
        self.security = security
        self.mcp_servers = mcp_servers
        self.logger = logger
        self.cost_tracker = cost_tracker


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
        interaction_level: str = "minimal",
        storage_dir: Optional[str] = None,
        clarification_prompt: Optional[str] = None,
        on_ask_user: Optional[Any] = None,
        allow_delegation: bool = False,
        delegation_timeout: int = 300,
        share_tools_with_agents: bool = False,
        max_tokens: int = 8096,
        recovery: Optional[dict | RecoveryManager] = None,
        security: Optional[dict | Sandbox] = None,
        mcp_servers: Optional[list[dict | MCPServer]] = None,
        logger: Optional[Logger] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        # Resolve provider only when the user explicitly passes provider or model.
        if provider is not None:
            self.provider = provider
        elif model:
            self.provider = resolve_provider(model=model, api_key=api_key, max_tokens=max_tokens)
        else:
            raise ValueError("Orchestrator requires either a model or provider.")

        # Store config values for internal use
        self._clarification_prompt = clarification_prompt
        self._on_ask_user = on_ask_user
        self._interaction_level = interaction_level
        self._allow_delegation = allow_delegation
        self._delegation_timeout = delegation_timeout
        self._share_tools_with_agents = share_tools_with_agents
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.recovery = self._build_recovery_manager(recovery)
        self.sandbox = self._build_sandbox(security)
        self.command_guard = (
            CommandGuard(self.sandbox)
            if self.sandbox and self.sandbox.enabled
            else None
        )
        self._mcp_server_configs = mcp_servers or []
        self.mcp_connector = MCPConnector()
        self._mcp_connected = False
        self._mcp_tool_names: set[str] = set()
        self._requested_prebuilt_tools = {
            tool for tool in (tools or []) if isinstance(tool, str)
        }

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
        self._base_tools = self.tool_registry.get_all()

        # MCP servers are connected lazily at run time because connection is async
        self._setup_mcp_servers(self._mcp_server_configs)

        # Agents: build lookup dict from agent.name
        self._agents: dict[str, Agent] = {a.name: a for a in (agents or [])}
        self._attach_cost_tracker_to_agents()
        if self._share_tools_with_agents:
            self._apply_shared_tools_to_agents()
        self._apply_security_to_agents()

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

        result = await agent.arun(task=task)
        return result.output if result.success else f"Error: {result.error}"

    # --- Agent management ---

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        if self.cost_tracker and not agent.cost_tracker:
            agent.cost_tracker = self.cost_tracker
        self._agents[agent.name] = agent
        if self._share_tools_with_agents:
            self._apply_shared_tools_to_agent(agent)
        self._apply_security_to_agent(agent)

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

    async def aresume(self) -> OrchestratorResult:
        """Resume from the latest recovery savepoint."""
        return await self._resume_async()

    async def arun_or_resume(self, prompt: str) -> OrchestratorResult:
        """
        Async entry point for apps that should continue unfinished work automatically.

        If recovery has an unfinished savepoint, Veska resumes it. Otherwise it
        starts a fresh run with the given prompt.
        """
        save_point = self.recovery.load_latest() if self.recovery else None
        if save_point and self._savepoint_has_unfinished_work(save_point):
            self._log_info(
                "Unfinished recovery savepoint found",
                {"savepoint_id": save_point.id, "stage": save_point.metadata.get("stage")},
            )
            return await self._resume_async()

        return await self._run_async(prompt)

    def run(self, prompt: str) -> OrchestratorResult:
        """
        Sync entry point for scripts. Do NOT call inside a running event loop.
        Use: result = await orchestrator.arun(...) instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "orchestrator.run() was called inside a running event loop. "
                "Use: result = await orchestrator.arun(...) instead."
            )

        return asyncio.run(self.arun(prompt))

    def run_or_resume(self, prompt: str) -> OrchestratorResult:
        """
        Sync entry point for apps that should continue unfinished work automatically.
        Do NOT call inside a running event loop.
        Use: result = await orchestrator.arun_or_resume(...) instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "orchestrator.run_or_resume() was called inside a running event loop. "
                "Use: result = await orchestrator.arun_or_resume(...) instead."
            )

        return asyncio.run(self.arun_or_resume(prompt))

    def resume(self) -> OrchestratorResult:
        """
        Sync resume entry point. Do NOT call inside a running event loop.
        Use: result = await orchestrator.aresume(...) instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "orchestrator.resume() was called inside a running event loop. "
                "Use: result = await orchestrator.aresume() instead."
            )

        return asyncio.run(self.aresume())

    async def _run_async(self, prompt: str) -> OrchestratorResult:
        """Internal async implementation."""
        self._status = "planning"
        self._cancelled = False
        self._paused = False
        self._log_info("Run started", {"prompt": prompt[:200]})

        await self.events.emit(Event(
            type=EventType.STARTED,
            source="orchestrator",
            message=f"Starting: {prompt[:100]}",
        ))

        try:
            await self._ensure_mcp_connected()

            # Step 0: Clarify (if developer set clarification_prompt)
            clarifications = await self._clarify(prompt)
            if clarifications:
                self._log_info("Clarifications received")
                prompt = prompt + clarifications

            # Step 1: Create the plan
            self._log_info("Planning started")
            plan = await self._create_plan(prompt)
            if not plan:
                self._log_error("Planning failed")
                return OrchestratorResult(
                    success=False, error="Failed to create plan"
                )

            self._current_plan = plan
            self._log_info("Plan created", {"tasks": self._count_plan_tasks(plan)})

            # Step 2: Checkpoint - show plan to user
            if self._interaction_level != "minimal":
                self._log_info("Checkpoint started", {"checkpoint_id": "plan_review"})
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
                        self._log_error("Plan rejected by user")
                        return OrchestratorResult(
                            success=False, error="Plan rejected by user"
                        )

            # Step 3: Build tasks from plan
            self._build_tasks_from_plan(plan)
            self._save_recovery_point("plan_created", {"prompt": prompt[:200]})

            # Step 4: Execute tasks
            self._status = "running"
            await self._execute_tasks()

            # Step 5: Collect results
            results = self._collect_results()

            self._status = "done"
            self._log_info("Run completed", self.task_planner.progress)
            self._save_recovery_point("run_completed", {"progress": self.task_planner.progress})
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
            self._log_error("Run failed", {"error": str(e)})
            await self.events.emit_error("orchestrator", str(e))
            return OrchestratorResult(success=False, error=str(e))

    async def _resume_async(self) -> OrchestratorResult:
        """Internal async resume implementation."""
        if not self.recovery:
            return OrchestratorResult(success=False, error="No recovery manager configured")

        save_point = self.recovery.load_latest()
        if not save_point:
            return OrchestratorResult(success=False, error="No recovery savepoint found")

        self._status = "running"
        self._cancelled = False
        self._paused = False
        self._log_info("Resume started", {"savepoint_id": save_point.id})

        try:
            await self._ensure_mcp_connected()
            self._restore_from_savepoint(save_point)
            RecoveryManager.get_resumable_tasks(self.task_planner)
            self._save_recovery_point(
                "resume_started",
                {"savepoint_id": save_point.id, "from_stage": save_point.metadata.get("stage")},
            )

            await self._execute_tasks()
            results = self._collect_results()

            self._status = "done"
            self._log_info("Resume completed", self.task_planner.progress)
            self._save_recovery_point("resume_completed", {"progress": self.task_planner.progress})

            await self.events.emit(Event(
                type=EventType.COMPLETED,
                source="orchestrator",
                message="Resume completed",
                data=results,
            ))

            return OrchestratorResult(
                success=not self.task_planner.has_failures,
                plan=save_point.plan_data,
                results=results,
                progress=self.task_planner.progress,
            )

        except Exception as e:
            self._status = "failed"
            self._log_error("Resume failed", {"error": str(e)})
            await self.events.emit_error("orchestrator", str(e))
            return OrchestratorResult(success=False, error=str(e))

    async def _create_plan(self, prompt: str) -> Optional[dict]:
        """Use AI to break the user's prompt into a structured plan."""
        if not self.provider:
            return None

        planning_prompt = self._build_planning_prompt(prompt)

        messages = [
            Message(role="system", content=planning_prompt),
            Message(role="user", content=prompt),
        ]

        response = await self.provider.chat(
            messages=messages,
            thinking=self._thinking.get_config() if self._thinking.enabled else None,
        )
        self._track_cost(response, "planning")

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
        self._log_info(
            "Task started",
            {"task_id": task.id, "task_name": task.name, "agent": task.agent},
        )
        self._save_recovery_point(
            "task_started",
            {"task_id": task.id, "task_name": task.name, "agent": task.agent},
        )

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
            result = await agent.arun(
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
                self._log_info(
                    "Task completed",
                    {"task_id": task.id, "task_name": task.name, "agent": task.agent},
                )
                self._save_recovery_point(
                    "task_completed",
                    {"task_id": task.id, "task_name": task.name, "agent": task.agent},
                )
            else:
                self.task_planner.fail_task(task.id, result.error or "Unknown error")
                self._log_error(
                    "Task failed",
                    {
                        "task_id": task.id,
                        "task_name": task.name,
                        "agent": task.agent,
                        "error": result.error,
                    },
                )

                await self.events.emit(Event(
                    type=EventType.TASK_FAILED,
                    source=task.agent,
                    message=f"Failed: {task.name} - {result.error}",
                    data={"task_id": task.id, "error": result.error},
                ))
                self._save_recovery_point(
                    "task_failed",
                    {
                        "task_id": task.id,
                        "task_name": task.name,
                        "agent": task.agent,
                        "error": result.error,
                    },
                )

                # Auto-retry if possible
                if self.task_planner.get_task(task.id).can_retry:
                    self.task_planner.retry_task(task.id)

        except Exception as e:
            self.task_planner.fail_task(task.id, str(e))
            self._log_error(
                "Task crashed",
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "agent": task.agent,
                    "error": str(e),
                },
            )
            await self.events.emit_error(task.agent, f"Task '{task.name}' crashed: {e}")
            self._save_recovery_point(
                "task_crashed",
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "agent": task.agent,
                    "error": str(e),
                },
            )

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

    def _log_info(self, message: str, data: Optional[dict] = None) -> None:
        if self.logger:
            self.logger.info("orchestrator", message, data)

    def _log_error(self, message: str, data: Optional[dict] = None) -> None:
        if self.logger:
            self.logger.error("orchestrator", message, data)

    def _count_plan_tasks(self, plan: dict) -> int:
        return sum(len(phase.get("tasks", [])) for phase in plan.get("phases", []))

    def _setup_mcp_servers(self, server_configs: list[dict | MCPServer]) -> None:
        for config in server_configs:
            if isinstance(config, MCPServer):
                self.mcp_connector.add_server(config)
            elif isinstance(config, dict):
                self.mcp_connector.add_server(MCPServer(**config))
            else:
                raise TypeError(f"Expected MCPServer or dict, got {type(config)}")

    async def _ensure_mcp_connected(self) -> None:
        if self._mcp_connected or not self._mcp_server_configs:
            return

        results = await self.mcp_connector.connect_all()
        connected = [name for name, ok in results.items() if ok]
        failed = [name for name, ok in results.items() if not ok]

        if connected:
            self._log_info("MCP servers connected", {"servers": connected})
        if failed:
            self._log_error("MCP servers failed to connect", {"servers": failed})

        mcp_tools = self.mcp_connector.get_all_tools()
        for tool in mcp_tools:
            if not self.tool_registry.has(tool.name):
                self.tool_registry.register(tool)
                self._mcp_tool_names.add(tool.name)

        if self._share_tools_with_agents:
            self._apply_shared_tools_to_agents()
        self._mcp_connected = True

    def _apply_shared_tools_to_agents(self) -> None:
        shared_tools = self.tool_registry.get_all()
        if not shared_tools:
            return

        for agent in self._agents.values():
            self._apply_shared_tools_to_agent(agent)

    def _apply_shared_tools_to_agent(self, agent: Agent) -> None:
        shared_tools = self.tool_registry.get_all()
        if not shared_tools:
            return

        existing = {tool.name for tool in agent.tools}
        tools_to_add = [tool for tool in shared_tools if tool.name not in existing]
        if tools_to_add:
            agent.update_tools(agent.tools + tools_to_add)

    def _track_cost(self, response: Any, task_id: str) -> None:
        if not self.cost_tracker:
            return
        self.cost_tracker.record(
            agent_name="orchestrator",
            model=response.model or (self.provider.model if self.provider else ""),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            task_id=task_id,
        )

    def _attach_cost_tracker_to_agents(self) -> None:
        if not self.cost_tracker:
            return
        for agent in self._agents.values():
            if not agent.cost_tracker:
                agent.cost_tracker = self.cost_tracker

    def _build_sandbox(self, security: Optional[dict | Sandbox]) -> Optional[Sandbox]:
        if security is None:
            return None

        if isinstance(security, Sandbox):
            return security

        if not isinstance(security, dict):
            raise TypeError(f"Expected dict or Sandbox, got {type(security)}")

        if not security.get("enabled", False):
            return None

        project_root = security.get("project_root") or Path.cwd()
        sandbox = Sandbox(
            project_root=str(project_root),
            framework_root=security.get("framework_root"),
        )

        territories = security.get("territories") or {}
        read_access = security.get("read_access") or {}
        for agent_name, territory_config in territories.items():
            if isinstance(territory_config, dict):
                territory = territory_config.get("path") or territory_config.get("territory")
                agent_read_access = territory_config.get("read_access")
            else:
                territory = territory_config
                agent_read_access = read_access.get(agent_name)

            if territory:
                sandbox.set_territory(
                    agent_name,
                    str(territory),
                    read_access=agent_read_access,
                )

        return sandbox

    def _apply_security_to_agents(self) -> None:
        if not self.sandbox:
            return
        for agent in self._agents.values():
            self._apply_security_to_agent(agent)

    def _apply_security_to_agent(self, agent: Agent) -> None:
        if not self.sandbox:
            return

        tools = list(agent.tools)
        existing_names = {tool.name for tool in tools}

        if (
            (
                self._share_tools_with_agents
                and "code_runner" in self._requested_prebuilt_tools
            )
            or self._has_code_runner_tools(existing_names)
        ):
            tools = self._replace_tool_group(
                tools,
                self._code_runner_tool_names(),
                get_code_runner_tools(self.command_guard, agent.name),
            )

        if (
            (
                self._share_tools_with_agents
                and "file_manager" in self._requested_prebuilt_tools
            )
            or self._has_file_manager_tools(existing_names)
        ):
            tools = self._replace_tool_group(
                tools,
                self._file_manager_tool_names(),
                get_file_manager_tools(self.sandbox, agent.name),
            )

        agent.update_tools(tools)

    def _replace_tool_group(
        self,
        tools: list[Tool],
        tool_names: set[str],
        replacements: list[Tool],
    ) -> list[Tool]:
        kept_tools = [tool for tool in tools if tool.name not in tool_names]
        kept_names = {tool.name for tool in kept_tools}
        return kept_tools + [
            tool for tool in replacements if tool.name not in kept_names
        ]

    def _has_code_runner_tools(self, tool_names: set[str]) -> bool:
        return bool(tool_names & self._code_runner_tool_names())

    def _has_file_manager_tools(self, tool_names: set[str]) -> bool:
        return bool(tool_names & self._file_manager_tool_names())

    def _code_runner_tool_names(self) -> set[str]:
        return {
            "run_python",
            "run_node",
            "run_command",
            "install_package",
            "run_tests",
        }

    def _file_manager_tool_names(self) -> set[str]:
        return {
            "create_file",
            "read_file",
            "edit_file",
            "delete_file",
            "list_files",
            "search_files",
        }

    def _build_recovery_manager(
        self, recovery: Optional[dict | RecoveryManager]
    ) -> Optional[RecoveryManager]:
        if recovery is None:
            return None

        if isinstance(recovery, RecoveryManager):
            return recovery

        if not isinstance(recovery, dict):
            raise TypeError(f"Expected dict or RecoveryManager, got {type(recovery)}")

        manager = RecoveryManager(enabled=bool(recovery.get("enabled", False)))
        storage_dir = recovery.get("storage_dir")
        if storage_dir:
            manager.set_storage_dir(storage_dir)
        save_callback = recovery.get("save_callback")
        if save_callback:
            manager.set_save_callback(save_callback)
        load_callback = recovery.get("load_callback")
        if load_callback:
            manager.set_load_callback(load_callback)
        return manager

    def _save_recovery_point(
        self, stage: str, metadata: Optional[dict] = None
    ) -> None:
        if not self.recovery:
            return

        save_point = self.recovery.save(
            plan_data=self.task_planner.to_dict(),
            task_states=self._get_task_states(),
            agent_memories=self._get_agent_memories(),
            shared_memory=self.shared_memory.to_dict(),
            metadata={"stage": stage, **(metadata or {})},
        )

        if save_point:
            self._log_info(
                "Recovery savepoint created",
                {"savepoint_id": save_point.id, "stage": stage},
            )

    def _get_task_states(self) -> dict[str, dict]:
        return {
            task.id: task.model_dump()
            for task in self.task_planner.get_all_tasks()
        }

    def _get_agent_memories(self) -> dict[str, dict]:
        return {
            name: agent.memory.to_dict()
            for name, agent in self._agents.items()
        }

    def _restore_from_savepoint(self, save_point: SavePoint) -> None:
        self.task_planner = TaskPlanner.from_dict(save_point.plan_data)
        self.shared_memory = SharedMemory.from_dict(save_point.shared_memory)

        for name, memory_data in save_point.agent_memories.items():
            agent = self._agents.get(name)
            if not agent:
                continue
            agent.memory = AgentMemory.from_dict(memory_data)
            agent._status = agent.memory.current_state
            self.shared_memory.store(agent.memory)

        self._current_plan = save_point.plan_data
        self._log_info(
            "Recovery savepoint restored",
            {"savepoint_id": save_point.id, "stage": save_point.metadata.get("stage")},
        )

    def _savepoint_has_unfinished_work(self, save_point: SavePoint) -> bool:
        planner = TaskPlanner.from_dict(save_point.plan_data)
        return any(
            task.status not in (TaskStatus.DONE, TaskStatus.CANCELLED)
            for task in planner.get_all_tasks()
        )


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
