"""Tests for automatic logging integration."""

import asyncio
import json

from veska.core.agent import Agent
from veska.core.orchestrator import Orchestrator
from veska.logging.logger import Logger
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig


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
        return ProviderResponse(
            content=self.output,
            input_tokens=10,
            output_tokens=5,
            model=self.model,
        )

    @property
    def provider_name(self) -> str:
        return "test"

    def supports_thinking(self) -> bool:
        return False


class PlanningProvider(StaticProvider):
    def __init__(self, plan: dict):
        super().__init__(json.dumps(plan), model="test-planner")


def test_agent_logs_important_events_when_logger_enabled():
    logger = Logger(enabled=True)
    agent = Agent(name="assistant", provider=StaticProvider("done"), logger=logger)

    result = agent.run("Say done")

    messages = [entry.message for entry in logger.get_logs()]
    assert result.success
    assert "Agent started" in messages
    assert "Model response received" in messages
    assert "Agent completed" in messages


def test_orchestrator_logs_important_events_when_logger_enabled():
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
    logger = Logger(enabled=True)
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(plan),
        agents=[agent],
        logger=logger,
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    messages = [entry.message for entry in logger.get_logs()]
    assert result.success
    assert "Run started" in messages
    assert "Planning started" in messages
    assert "Plan created" in messages
    assert "Task started" in messages
    assert "Task completed" in messages
    assert "Run completed" in messages
