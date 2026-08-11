# Veska

![CI](https://github.com/NavedAlam0/Veska/actions/workflows/ci.yml/badge.svg)

Veska is a Python framework for building agent and multi-agent workflows with tools, MCP, logging, cost tracking, recovery, and security controls.

## Install

```bash
pip install veska
```

## Quick Start

```python
from veska import Agent, Orchestrator

frontend = Agent(
    name="frontend",
    system_prompt="Build frontend UI.",
    model="gpt-4o",
    tools=["file_manager"],
)

backend = Agent(
    name="backend",
    system_prompt="Build backend APIs.",
    model="gpt-4o",
    tools=["file_manager", "code_runner"],
)

orchestrator = Orchestrator(
    model="gpt-4o",
    agents=[frontend, backend],
)

result = orchestrator.run("Build a small blog app")
print(result.results)
```

## Tools

Tools passed to an agent belong to that agent:

```python
agent = Agent(
    model="gpt-4o",
    tools=["file_manager", "code_runner", "code_scanner"],
)
```

Tools passed to the orchestrator stay with the orchestrator by default:

```python
orchestrator = Orchestrator(
    model="gpt-4o",
    tools=["code_scanner"],
)
```

To share orchestrator tools and orchestrator MCP tools with agents:

```python
orchestrator = Orchestrator(
    tools=["code_scanner"],
    mcp_servers=[github],
    share_tools_with_agents=True,
)
```

## MCP

MCP can be connected directly to an agent or to the orchestrator:

```python
from veska import MCPServer

github = MCPServer(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "..."},
)

agent = Agent(model="gpt-4o", mcp_servers=[github])
orchestrator = Orchestrator(model="gpt-4o", mcp_servers=[github])
```

## Logging And Cost Tracking

Logging and cost tracking are explicit objects:

```python
from veska import CostTracker, Logger

logger = Logger(enabled=True)
cost_tracker = CostTracker(enabled=True)

orchestrator = Orchestrator(
    model="gpt-4o",
    logger=logger,
    cost_tracker=cost_tracker,
)
```

## Recovery

Use recovery when workflows should continue after interruption:

```python
from veska import RecoveryManager

recovery = RecoveryManager(enabled=True)

orchestrator = Orchestrator(
    model="gpt-4o",
    agents=[frontend, backend],
    recovery=recovery,
)

result = orchestrator.run_or_resume("Build a small blog app")
```

## Security

Security protects project boundaries and guards built-in file/command tools:

```python
orchestrator = Orchestrator(
    model="gpt-4o",
    agents=[frontend, backend],
    security={"enabled": True, "project_root": "/path/to/project"},
)
```

Territories are optional. Use them only when agents should be limited to separate folders.

## Media

Agents can receive attachments:

```python
from veska import Audio, Image, PDF

result = agent.run(
    "Answer using these files",
    attachments=[Audio("voice.mp3"), Image("screen.png"), PDF("brief.pdf")],
)
```

Audio is sent only through the selected provider/model. If that model does not support raw audio, Veska returns an error before calling the provider.

## Features

- Multi-agent orchestration with dependency-aware task execution
- Agent-level and orchestrator-level tools
- Agent-level and orchestrator-level MCP
- Optional structured logging and cost tracking
- Recovery savepoints with `run_or_resume`
- Security sandbox for built-in file and command tools
- Optional built-in `code_scanner` tool
- Image, PDF, and audio attachments
- Structured output, streaming, memory, and thinking support

## Requirements

- Python 3.10+
- `anthropic`, `openai`, `pydantic`
