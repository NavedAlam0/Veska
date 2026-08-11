"""Tests for automatic cost tracking integration."""

import asyncio
import json

from veska.core.agent import Agent
from veska.core.orchestrator import Orchestrator
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig
from veska.tracking.cost_tracker import CostTracker


class UsageProvider(BaseProvider):
    def __init__(
        self,
        output: str,
        model: str = "test-model",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        super().__init__(api_key="", model=model)
        self.output = output
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        thinking: ThinkingConfig | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse:
        return ProviderResponse(
            content=self.output,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model=self.model,
        )

    @property
    def provider_name(self) -> str:
        return "test"

    def supports_thinking(self) -> bool:
        return False


class PlanningProvider(UsageProvider):
    def __init__(self, plan: dict):
        super().__init__(
            output=json.dumps(plan),
            model="test-planner",
            input_tokens=20,
            output_tokens=10,
        )


def _tracker(enabled: bool = True) -> CostTracker:
    return CostTracker(
        enabled=enabled,
        pricing={
            "test-model": {"input": 1.0, "output": 2.0},
            "test-planner": {"input": 1.0, "output": 2.0},
        },
    )


def test_agent_records_cost_when_tracker_enabled():
    tracker = _tracker(enabled=True)
    agent = Agent(
        name="assistant",
        provider=UsageProvider("done"),
        cost_tracker=tracker,
    )

    result = agent.run("Say done")

    records = tracker.get_records()
    assert result.success
    assert len(records) == 1
    assert records[0].agent_name == "assistant"
    assert records[0].input_tokens == 100
    assert records[0].output_tokens == 50
    assert tracker.total_tokens == {"input": 100, "output": 50, "total": 150}
    assert tracker.total_cost == 0.0002


def test_agent_does_not_record_cost_when_tracker_disabled():
    tracker = _tracker(enabled=False)
    agent = Agent(
        name="assistant",
        provider=UsageProvider("done"),
        cost_tracker=tracker,
    )

    result = agent.run("Say done")

    assert result.success
    assert tracker.get_records() == []
    assert tracker.total_tokens == {"input": 0, "output": 0, "total": 0}


def test_orchestrator_tracks_planning_and_agent_usage_with_shared_tracker():
    plan = {
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
    tracker = _tracker(enabled=True)
    agent = Agent(name="worker", provider=UsageProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(plan),
        agents=[agent],
        cost_tracker=tracker,
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    records = tracker.get_records()
    assert result.success
    assert len(records) == 2
    assert [record.agent_name for record in records] == ["orchestrator", "worker"]
    assert tracker.total_tokens == {"input": 120, "output": 60, "total": 180}
