# Veska

A multi-agent AI framework built from scratch in Python. No LangChain, no CrewAI, no AutoGen — 100% custom-built for full control.

## Install

```bash
pip install veska
```

## Quick Start

```python
from veska import Agent

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful coding assistant.",
    model="claude-sonnet-4-6",
)

result = agent.run("Explain what a decorator is in Python")
print(result.output)
```

## Add Tools

```python
from veska import Agent, tool

@tool
def get_weather(city: str):
    return f"Weather in {city}: 72°F, sunny"

agent = Agent(
    name="weather-bot",
    system_prompt="You help users check the weather. Use the get_weather tool.",
    model="claude-sonnet-4-6",
    tools=[get_weather],
)

result = agent.run("What's the weather in Paris?")
print(result.output)
```

## Structured Output

```python
from veska import Agent

agent = Agent(
    name="reviewer",
    system_prompt="You review movies.",
    model="claude-sonnet-4-6",
    output_format={
        "title": str,
        "rating": float,
        "recommend": bool,
    }
)

result = agent.run("Review the movie Inception")
print(result.output["title"])      # "Inception"
print(result.output["rating"])     # 9.0
```

## Streaming

```python
result = agent.run("Write a haiku about coding", stream=True)
```

## Multi-Agent System

```python
from veska import Agent, Orchestrator

researcher = Agent(
    name="researcher",
    system_prompt="You research topics thoroughly.",
    model="claude-sonnet-4-6",
)

writer = Agent(
    name="writer",
    system_prompt="You write clear, engaging content.",
    model="claude-sonnet-4-6",
)

orchestrator = Orchestrator(
    model="claude-sonnet-4-6",
    agents=[researcher, writer],
    tools=["file_manager"],
)

result = orchestrator.run("Write a blog post about AI agents")
print(result.results)
```

## Per-Agent Models

```python
researcher = Agent(name="researcher", model="claude-sonnet-4-6")
writer = Agent(name="writer", model="gpt-4o")
```

## Features

- **Multi-Agent Orchestration** — Orchestrator breaks tasks into a dependency graph, runs them in parallel/sequential order
- **Multi-Model Support** — Claude and OpenAI, configurable per agent
- **Unified Tool System** — Pre-built, custom, and MCP tools. Just use `@tool` decorator
- **Streaming** — `stream=True` or `stream=callback`
- **Structured Output** — Pass `output_format` dict, get validated responses
- **Memory System** — Private memory per agent + shared memory pool
- **3-Level Error Recovery** — Auto-retry, agent-level fix, discussion room
- **Security Sandboxing** — Agents sandboxed to their own territory
- **Extended Thinking** — Optional per-agent thinking support
- **MCP Support** — Connect external services via Model Context Protocol
- **General Purpose** — Not locked to any use case

## Requirements

- Python 3.10+
- `anthropic`, `openai`, `pydantic`

## License

MIT
