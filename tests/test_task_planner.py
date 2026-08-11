"""Tests for the Task Planner — dependency graph and execution order."""

import asyncio
import json

from veska.core.agent import Agent
from veska.core.orchestrator import Orchestrator
from veska.core.task_planner import Task, TaskPlanner, TaskStatus
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig


class StaticProvider(BaseProvider):
    """Provider test double that avoids real API calls while using real Agent objects."""

    def __init__(self, output: str):
        super().__init__(api_key="", model="test-model")
        self.output = output

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        thinking: ThinkingConfig | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(content=self.output, model=self.model)

    @property
    def provider_name(self) -> str:
        return "test"

    def supports_thinking(self) -> bool:
        return False


class PlanningProvider(BaseProvider):
    """Returns a fixed plan for the orchestrator planning call."""

    def __init__(self, plan: dict):
        super().__init__(api_key="", model="test-planner")
        self.plan = plan

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        thinking: ThinkingConfig | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(content=json.dumps(self.plan), model=self.model)

    @property
    def provider_name(self) -> str:
        return "test"

    def supports_thinking(self) -> bool:
        return False


def test_independent_tasks_run_in_parallel():
    """Tasks with no dependencies should be in the same wave."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="r")
    p.add_tasks([a, b])

    waves = p.get_execution_order()
    assert len(waves) == 1
    assert set(waves[0]) == {"a", "b"}


def test_dependent_task_runs_after():
    """A task depending on others should be in a later wave."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="r")
    c = Task(id="c", name="C", agent="w", depends_on=["a", "b"])
    p.add_tasks([a, b, c])

    waves = p.get_execution_order()
    assert len(waves) == 2
    assert set(waves[0]) == {"a", "b"}
    assert waves[1] == ["c"]


def test_three_level_dependency_chain():
    """A → B → C should produce 3 waves."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
        Task(id="c", name="C", agent="r", depends_on=["b"]),
    ])

    waves = p.get_execution_order()
    assert len(waves) == 3
    assert waves[0] == ["a"]
    assert waves[1] == ["b"]
    assert waves[2] == ["c"]


def test_circular_dependency_detected():
    """Circular dependencies should be caught by validate()."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r", depends_on=["b"]),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
    ])

    errors = p.validate()
    assert len(errors) > 0
    assert "Circular" in errors[0]


def test_missing_dependency_detected():
    """A task depending on a non-existent task should be flagged."""
    p = TaskPlanner()
    p.add_task(Task(id="a", name="A", agent="r", depends_on=["missing"]))

    errors = p.validate()
    assert len(errors) > 0
    assert "missing" in errors[0]


def test_valid_plan_has_no_errors():
    """A valid plan should return empty errors list."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
    ])

    assert p.validate() == []


def test_get_ready_tasks():
    """Only tasks with all dependencies met should be ready."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="w", depends_on=["a"])
    p.add_tasks([a, b])

    ready = p.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "a"


def test_completing_task_unlocks_dependents():
    """After completing a task, its dependents should become ready."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="w", depends_on=["a"])
    p.add_tasks([a, b])

    p.complete_task("a", "done")
    ready = p.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "b"


def test_task_retry():
    """A failed task should be retryable up to max_retries."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r", max_retries=2)
    p.add_task(a)

    p.fail_task("a", "error")
    assert p.get_task("a").can_retry

    p.retry_task("a")
    assert p.get_task("a").status == TaskStatus.RETRYING
    assert p.get_task("a").retries == 1


def test_progress_tracking():
    """Progress should reflect task statuses."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r"),
    ])

    assert p.progress["total"] == 2
    assert p.progress["completed"] == 0

    p.complete_task("a", "done")
    assert p.progress["completed"] == 1
    assert p.progress["percentage"] == 50.0


def test_is_complete():
    """is_complete should be True only when all tasks are terminal."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r"),
    ])

    assert not p.is_complete
    p.complete_task("a", "done")
    assert not p.is_complete
    p.complete_task("b", "done")
    assert p.is_complete


def test_orchestrator_runs_agent_with_async_entrypoint():
    """Orchestrator must call arun() because _run_task runs inside async code."""

    agent = Agent(name="worker", provider=StaticProvider("done: Build feature"))
    orchestrator = Orchestrator(provider=object())
    orchestrator.register_agent(agent)

    task = Task(id="t1", name="Build", description="Build feature", agent=agent.name)
    orchestrator.task_planner.add_task(task)

    asyncio.run(orchestrator._run_task(task))

    assert orchestrator.task_planner.get_task("t1").status == TaskStatus.DONE
    assert orchestrator.task_planner.get_task("t1").result == "done: Build feature"


def test_orchestrator_delegation_uses_async_entrypoint():
    """Delegated work must also call arun(), not the sync run() wrapper."""

    agent = Agent(name="delegate", provider=StaticProvider("delegated: Handle subtask"))
    orchestrator = Orchestrator(provider=object())
    orchestrator.register_agent(agent)

    result = asyncio.run(
        orchestrator._run_delegate(agent_name=agent.name, task="Handle subtask", depth=0)
    )

    assert result == "delegated: Handle subtask"


def test_orchestrator_default_interaction_level_does_not_wait_for_checkpoint():
    """Default orchestrator runs should not hang waiting for plan approval."""

    plan = {
        "name": "Test Plan",
        "description": "Run one task",
        "phases": [
            {
                "name": "Execution",
                "description": "Execute task",
                "tasks": [
                    {
                        "id": "main",
                        "name": "Main task",
                        "description": "Do the work",
                        "agent": "worker",
                        "depends_on": [],
                    }
                ],
            }
        ],
    }
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(provider=PlanningProvider(plan), agents=[agent])

    result = asyncio.run(
        asyncio.wait_for(orchestrator.arun("Do the work"), timeout=1)
    )

    assert result.success
    assert result.progress["completed"] == 1
    assert orchestrator.events.stats["pending_checkpoints"] == 0
