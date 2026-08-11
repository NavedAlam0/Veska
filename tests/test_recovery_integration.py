"""Tests for orchestrator recovery savepoint integration."""

import asyncio
import json

from veska.core.agent import Agent
from veska.core.orchestrator import Orchestrator
from veska.core.task_planner import Task, TaskPlanner
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig
from veska.recovery.recovery import RecoveryManager


class StaticProvider(BaseProvider):
    def __init__(self, output: str, model: str = "test-model"):
        super().__init__(api_key="", model=model)
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


class PlanningProvider(StaticProvider):
    def __init__(self, plan: dict):
        super().__init__(json.dumps(plan), model="test-planner")


class CountingProvider(StaticProvider):
    def __init__(self, output: str):
        super().__init__(output)
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        thinking: ThinkingConfig | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse:
        self.calls += 1
        return await super().chat(messages, tools, thinking, stream, **kwargs)


def _plan() -> dict:
    return {
        "name": "Test Plan",
        "description": "Run one task",
        "phases": [
            {
                "name": "Execution",
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


def test_orchestrator_creates_recovery_savepoints_when_enabled():
    recovery = RecoveryManager(enabled=True)
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        recovery=recovery,
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    savepoints = recovery.get_save_points()
    stages = [sp.metadata["stage"] for sp in savepoints]
    assert result.success
    assert stages == ["plan_created", "task_started", "task_completed", "run_completed"]
    assert savepoints[-1].task_states["main"]["status"] == "done"
    assert "worker" in savepoints[-1].agent_memories


def test_orchestrator_recovery_dict_writes_latest_savepoint(tmp_path):
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        recovery={"enabled": True, "storage_dir": str(tmp_path)},
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    latest = tmp_path / "latest.json"
    data = json.loads(latest.read_text())
    assert result.success
    assert latest.exists()
    assert data["metadata"]["stage"] == "run_completed"
    assert data["task_states"]["main"]["status"] == "done"


def test_orchestrator_recovery_disabled_does_not_save():
    recovery = RecoveryManager(enabled=False)
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        recovery=recovery,
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    assert result.success
    assert recovery.get_save_points() == []


def test_orchestrator_resume_continues_unfinished_tasks_from_savepoint():
    recovery = RecoveryManager(enabled=True)
    planner = TaskPlanner()
    planner.add_tasks([
        Task(id="first", name="First", description="Already done", agent="worker"),
        Task(
            id="second",
            name="Second",
            description="Continue here",
            agent="worker",
            depends_on=["first"],
        ),
    ])
    planner.add_phase("Execution", ["first", "second"])
    planner.complete_task("first", "first result")
    planner.start_task("second")

    recovery.save(
        plan_data=planner.to_dict(),
        task_states={task.id: task.model_dump() for task in planner.get_all_tasks()},
        agent_memories={},
        shared_memory={},
        metadata={"stage": "task_started", "task_id": "second"},
    )

    provider = CountingProvider("second result")
    agent = Agent(name="worker", provider=provider)
    orchestrator = Orchestrator(provider=object(), agents=[agent], recovery=recovery)

    result = orchestrator.resume()

    tasks = result.results["tasks"]
    assert result.success
    assert provider.calls == 1
    assert tasks["first"]["status"] == "done"
    assert tasks["first"]["result"] == "first result"
    assert tasks["second"]["status"] == "done"
    assert tasks["second"]["result"] == "second result"


def test_orchestrator_resume_requires_savepoint():
    orchestrator = Orchestrator(provider=object(), recovery=RecoveryManager(enabled=True))

    result = orchestrator.resume()

    assert not result.success
    assert result.error == "No recovery savepoint found"


def test_orchestrator_run_or_resume_starts_fresh_without_savepoint():
    recovery = RecoveryManager(enabled=True)
    planner_provider = CountingProvider(json.dumps(_plan()))
    worker_provider = CountingProvider("fresh result")
    agent = Agent(name="worker", provider=worker_provider)
    orchestrator = Orchestrator(
        provider=planner_provider,
        agents=[agent],
        recovery=recovery,
    )

    result = orchestrator.run_or_resume("Do the work")

    assert result.success
    assert planner_provider.calls == 1
    assert worker_provider.calls == 1
    assert result.results["tasks"]["main"]["result"] == "fresh result"


def test_orchestrator_run_or_resume_continues_unfinished_savepoint():
    recovery = RecoveryManager(enabled=True)
    planner = TaskPlanner()
    planner.add_tasks([
        Task(id="first", name="First", description="Already done", agent="worker"),
        Task(
            id="second",
            name="Second",
            description="Continue here",
            agent="worker",
            depends_on=["first"],
        ),
    ])
    planner.add_phase("Execution", ["first", "second"])
    planner.complete_task("first", "first result")
    planner.start_task("second")

    recovery.save(
        plan_data=planner.to_dict(),
        task_states={task.id: task.model_dump() for task in planner.get_all_tasks()},
        agent_memories={},
        shared_memory={},
        metadata={"stage": "task_started", "task_id": "second"},
    )

    planner_provider = CountingProvider(json.dumps(_plan()))
    worker_provider = CountingProvider("second result")
    agent = Agent(name="worker", provider=worker_provider)
    orchestrator = Orchestrator(
        provider=planner_provider,
        agents=[agent],
        recovery=recovery,
    )

    result = orchestrator.run_or_resume("Do the work")

    tasks = result.results["tasks"]
    assert result.success
    assert planner_provider.calls == 0
    assert worker_provider.calls == 1
    assert tasks["first"]["status"] == "done"
    assert tasks["first"]["result"] == "first result"
    assert tasks["second"]["status"] == "done"
    assert tasks["second"]["result"] == "second result"


def test_orchestrator_run_or_resume_starts_fresh_after_completed_savepoint():
    recovery = RecoveryManager(enabled=True)
    planner = TaskPlanner()
    planner.add_task(Task(id="old", name="Old task", description="Done", agent="worker"))
    planner.add_phase("Execution", ["old"])
    planner.complete_task("old", "old result")

    recovery.save(
        plan_data=planner.to_dict(),
        task_states={task.id: task.model_dump() for task in planner.get_all_tasks()},
        agent_memories={},
        shared_memory={},
        metadata={"stage": "run_completed"},
    )

    planner_provider = CountingProvider(json.dumps(_plan()))
    worker_provider = CountingProvider("new result")
    agent = Agent(name="worker", provider=worker_provider)
    orchestrator = Orchestrator(
        provider=planner_provider,
        agents=[agent],
        recovery=recovery,
    )

    result = orchestrator.run_or_resume("Do new work")

    assert result.success
    assert planner_provider.calls == 1
    assert worker_provider.calls == 1
    assert "main" in result.results["tasks"]
    assert result.results["tasks"]["main"]["result"] == "new result"
