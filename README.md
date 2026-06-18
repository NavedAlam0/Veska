# Veska

A multi-agent AI framework built from scratch in Python. No LangChain, no CrewAI, no AutoGen — 100% custom-built for full control.

Multiple AI agents collaborate on tasks using a message bus, dependency-based task planning, built-in security, and 3-level error recovery.

## Features

- **Multi-Agent Orchestration** — Orchestrator breaks tasks into a dependency graph, runs them in hybrid parallel/sequential order
- **Multi-Model Support** — Claude and OpenAI, configurable per agent. Swap models without changing agent code
- **Unified Tool System** — Pre-built, custom, and MCP tools in one flat list. Agent doesn't know the difference
- **Message Bus** — Any agent can talk to any agent. Orchestrator watches and controls routing
- **Memory System** — Private memory per agent + shared memory pool controlled by orchestrator
- **3-Level Error Recovery** — Simple retry → agent-level fix → discussion room where agents collaborate on complex bugs
- **Security Sandboxing** — 3 zones: framework (locked), project (agent territories), system (blocked)
- **Context Window Management** — 3 layers: active context, summaries, full disk storage
- **Event System** — Framework emits signals, any app (web/CLI/desktop) listens. Framework doesn't know about UI
- **Extended Thinking** — Optional per-agent thinking support for models that support it
- **MCP Support** — Plug-and-play external services via Model Context Protocol
- **General Purpose** — Not locked to any use case. Build code generators, support bots, data pipelines, anything

## Installation

```bash
pip install -e .
```

**Requirements:** Python 3.10+

**Dependencies:** `anthropic`, `openai`, `pydantic` — that's it.

## Quick Start

```python
import asyncio
from veska import Agent, AgentConfig, Orchestrator, OrchestratorConfig
from veska import ClaudeProvider, Tool, ToolParameter

# 1. Create a provider
provider = ClaudeProvider(api_key="your-key", model="claude-sonnet-4-6")

# 2. Create agents (generic — you define the purpose)
backend = Agent(AgentConfig(
    name="backend_dev",
    system_prompt="You are a senior Python backend developer.",
    provider=provider,
))

frontend = Agent(AgentConfig(
    name="frontend_dev",
    system_prompt="You are a React frontend developer.",
    provider=provider,
))

# 3. Create orchestrator
orch = Orchestrator(OrchestratorConfig(
    provider=provider,
    tools=["file_manager", "code_runner"],
    agents={"backend": backend, "frontend": frontend},
))

# 4. Run
result = asyncio.run(orch.run("Build a blog app with REST API and React frontend"))
print(result)
```

## Using OpenAI Instead

```python
from veska import OpenAIProvider

provider = OpenAIProvider(api_key="your-key", model="gpt-4o")

agent = Agent(AgentConfig(
    name="analyst",
    system_prompt="You are a data analyst.",
    provider=provider,
))
```

Mix and match — different agents can use different providers.

## Custom Tools

```python
from veska import Tool, ToolParameter

sms_tool = Tool(
    name="send_sms",
    description="Send an SMS message",
    when_to_use="When the task requires sending a text notification",
    parameters=[
        ToolParameter(name="phone", type="string", description="Phone number"),
        ToolParameter(name="message", type="string", description="Message text"),
    ],
    function=my_sms_function,
)

# Add to the same tools list as pre-built tools
orch = Orchestrator(OrchestratorConfig(
    tools=["file_manager", "code_runner", sms_tool],
    ...
))
```

## Pre-built Tools

Add by name — zero code needed:

| Tool Group | Tools | Add with |
|---|---|---|
| **File Manager** | create_file, read_file, edit_file, delete_file, list_files, search_files | `"file_manager"` |
| **Code Runner** | run_python, run_node, run_command, install_package, run_tests | `"code_runner"` |
| **Project Tools** | create_folder, create_project_structure, init_git, create_env_file | `"project_tools"` |

## Tool Permissions

Control which agent can use which tool:

```python
from veska import ToolPermissions

perms = ToolPermissions(registry)
perms.set("backend_dev", ["create_file", "read_file", "run_python"])
perms.set("frontend_dev", ["create_file", "read_file", "run_node"])
```

## Event System

Framework sends events, your app decides how to display them:

```python
from veska import EventType

# Listen to events
orch.events.on(EventType.PROGRESS, lambda e: print(f"Progress: {e.message}"))
orch.events.on(EventType.ERROR, lambda e: print(f"Error: {e.message}"))
orch.events.on(EventType.CHECKPOINT, handle_checkpoint)

# Checkpoints — pause for user approval
async def handle_checkpoint(event):
    print(f"Plan ready: {event.data}")
    orch.events.approve_checkpoint(event.data["checkpoint_id"])
```

## Security

Agents have full freedom inside their own territory. Blocked everywhere else:

```python
from veska import Sandbox

sandbox = Sandbox(project_root="/my-project", framework_root="/path/to/veska")
sandbox.set_territory("backend_dev", "/my-project/backend")
sandbox.set_territory("frontend_dev", "/my-project/frontend")

# backend_dev can write to /my-project/backend/ — ALLOWED
# backend_dev can write to /my-project/frontend/ — BLOCKED
# backend_dev can read from /my-project/frontend/ — ALLOWED
# Any agent accessing /etc/passwd — BLOCKED
```

## Error Recovery

Three severity levels, automatic escalation:

| Level | What | Action |
|---|---|---|
| **Level 1** | Tool failure, timeout | Auto-retry (up to 2 times) |
| **Level 2** | Code error, permission issue | Agent gets error context, tries different approach |
| **Level 3** | Cross-agent problem | Discussion room — agents collaborate to find the fix |

```python
from veska import ErrorDetector, DiscussionRoom, FixCoordinator

detector = ErrorDetector()
error = detector.detect("backend_dev", "Interface mismatch with frontend API")
# error.severity == LEVEL_3 (cross-agent → discussion room)
```

## Optional Systems

All OFF by default. Enable only what you need:

```python
from veska import Logger, LogLevel, CostTracker, RecoveryManager

# Logging
logger = Logger(enabled=True, level=LogLevel.INFO)

# Cost tracking (stores in YOUR database, not ours)
tracker = CostTracker(enabled=True)
tracker.set_storage(my_db_save_function)

# Crash recovery
recovery = RecoveryManager(enabled=True)
recovery.set_storage_dir("./save_points")
```

## Extended Thinking

For models that support step-by-step reasoning:

```python
agent = Agent(AgentConfig(
    name="architect",
    system_prompt="You are a system architect.",
    provider=provider,
    thinking={"enabled": True, "budget_tokens": 10000, "output": "log"},
))
```

Output modes: `"discard"` (default), `"log"` (save for debugging), `"expose"` (send through events)

## MCP Support

Connect external services — their tools appear like any other tool:

```python
from veska import MCPConnector, MCPServer

connector = MCPConnector()
connector.add_server(MCPServer(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "your-token"},
))

await connector.connect_all()
tools = connector.get_all_tools()  # Ready to register
```

## Project Structure

```
veska/
├── core/                # Brain — agent, orchestrator, message bus, events, planner
│   ├── agent.py
│   ├── orchestrator.py
│   ├── message_bus.py
│   ├── events.py
│   ├── task_planner.py
│   ├── memory.py
│   ├── context_manager.py
│   ├── prompt_manager.py
│   ├── thinking.py
│   └── mcp_connector.py
├── providers/           # AI model connections
│   ├── base.py
│   ├── claude_provider.py
│   └── openai_provider.py
├── tools/               # Unified tool system
│   ├── base.py
│   ├── registry.py
│   ├── permissions.py
│   ├── file_manager.py
│   ├── code_runner.py
│   └── project_tools.py
├── security/            # Sandboxing & safety
│   ├── sandbox.py
│   ├── command_guard.py
│   └── code_scanner.py
├── recovery/            # Error handling & crash recovery
│   ├── error_recovery.py
│   └── recovery.py
├── logging/             # Optional structured logging
│   └── logger.py
└── tracking/            # Optional cost tracking
    └── cost_tracker.py
```

## Design Principles

1. **General purpose** — Framework works for any use case, not just code generation
2. **Simple by default** — Optional systems are OFF. Zero overhead until you need them
3. **Unified interfaces** — Tools, providers, and memory all use single consistent patterns
4. **No vendor lock-in** — Swap Claude for OpenAI (or both) without changing agent code
5. **Built from scratch** — 3 dependencies total. No hidden complexity, no framework churn
6. **Security first** — Agents are sandboxed. Can't touch framework code or system files

## License

MIT
