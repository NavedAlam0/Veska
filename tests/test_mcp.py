"""Tests for MCP orchestrator integration."""

import asyncio
import json

from veska.core.agent import Agent
from veska.core.mcp_connector import MCPServer
from veska.core.orchestrator import Orchestrator
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig
from veska.tools.base import Tool


class PlanningProvider(BaseProvider):
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


class StaticProvider(BaseProvider):
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


class FakeMCPServer(MCPServer):
    def __init__(self, name: str, tools: list[Tool], should_connect: bool = True):
        super().__init__(name=name, command="fake")
        self._provided_tools = tools
        self.should_connect = should_connect
        self.connect_called = False

    async def connect(self) -> bool:
        self.connect_called = True
        self._connected = self.should_connect
        self._tools = list(self._provided_tools) if self.should_connect else []
        return self.should_connect


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


def test_orchestrator_mcp_tools_are_not_shared_with_agents_by_default():
    mcp_tool = Tool(
        name="fake__lookup",
        description="Lookup from fake MCP",
        function=lambda query: f"found {query}",
    )
    server = FakeMCPServer("fake", [mcp_tool])
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        mcp_servers=[server],
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    assert result.success
    assert server.connect_called
    assert orchestrator.mcp_connector.connected_count == 1
    assert orchestrator.tool_registry.has("fake__lookup")
    assert "fake__lookup" not in {tool.name for tool in agent.tools}


def test_orchestrator_mcp_tools_can_be_shared_with_agents():
    mcp_tool = Tool(
        name="fake__lookup",
        description="Lookup from fake MCP",
        function=lambda query: f"found {query}",
    )
    server = FakeMCPServer("fake", [mcp_tool])
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        mcp_servers=[server],
        share_tools_with_agents=True,
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    assert result.success
    assert server.connect_called
    assert orchestrator.mcp_connector.connected_count == 1
    assert orchestrator.tool_registry.has("fake__lookup")
    assert "fake__lookup" in {tool.name for tool in agent.tools}


def test_orchestrator_logs_failed_mcp_connections_without_failing_run():
    server = FakeMCPServer("fake", [], should_connect=False)
    agent = Agent(name="worker", provider=StaticProvider("task complete"))
    orchestrator = Orchestrator(
        provider=PlanningProvider(_plan()),
        agents=[agent],
        mcp_servers=[server],
    )

    result = asyncio.run(orchestrator.arun("Do the work"))

    assert result.success
    assert server.connect_called
    assert orchestrator.mcp_connector.connected_count == 0


def test_agent_connects_mcp_servers_and_adds_tools_directly():
    mcp_tool = Tool(
        name="fake__lookup",
        description="Lookup from fake MCP",
        function=lambda query: f"found {query}",
    )
    server = FakeMCPServer("fake", [mcp_tool])
    agent = Agent(
        name="assistant",
        provider=StaticProvider("task complete"),
        mcp_servers=[server],
    )

    result = agent.run("Use available tools")

    assert result.success
    assert server.connect_called
    assert agent.mcp_connector.connected_count == 1
    assert "fake__lookup" in {tool.name for tool in agent.tools}


def test_agent_failed_mcp_connection_does_not_fail_run():
    server = FakeMCPServer("fake", [], should_connect=False)
    agent = Agent(
        name="assistant",
        provider=StaticProvider("task complete"),
        mcp_servers=[server],
    )

    result = agent.run("Continue without MCP")

    assert result.success
    assert server.connect_called
    assert agent.mcp_connector.connected_count == 0
