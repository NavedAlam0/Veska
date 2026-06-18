"""
MCP (Model Context Protocol) Connector for Veska (Optional).

Plug-and-play external services. Connect to any MCP server
and its tools appear in the agent's tool list like any other tool.

OFF by default. Developer adds MCP servers if they want them.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any, Optional

from veska.core.env import resolve_env_vars
from veska.tools.base import Tool, ToolParameter, ToolResult


class MCPServer:
    """
    Represents a connected MCP server.

    Each server provides tools that get registered into the
    agent's tool list alongside pre-built and custom tools.

    env accepts either:
      - list of env variable names: ["GITHUB_TOKEN"]
        → auto-fetches values from .env
      - dict of key-value pairs: {"GITHUB_TOKEN": "ghp-abc123"}
        → uses values directly
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[list[str] | dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []

        # Resolve env: list of names → auto-fetch from .env
        if isinstance(env, list):
            self.env = resolve_env_vars(env)
        else:
            self.env = env or {}
        self._process: Optional[asyncio.subprocess.Process] = None
        self._tools: list[Tool] = []
        self._connected = False
        self._request_id = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Starts the server process and discovers available tools.
        """
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict(__import__("os").environ), **self.env} if self.env else None,
            )

            # Initialize the connection
            init_response = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "veska", "version": "0.1.0"},
            })

            if init_response and "result" in init_response:
                # Send initialized notification
                await self._send_notification("notifications/initialized", {})

                # Discover tools
                await self._discover_tools()
                self._connected = True
                return True

            return False

        except (FileNotFoundError, OSError):
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
        self._connected = False
        self._tools.clear()

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server."""
        if not self._connected:
            return {"error": "Not connected to MCP server"}

        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        if response and "result" in response:
            content = response["result"].get("content", [])
            # Extract text content
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else str(content)

        error = response.get("error", {}).get("message", "Unknown error") if response else "No response"
        return {"error": error}

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        response = await self._send_request("tools/list", {})

        if not response or "result" not in response:
            return

        tools_data = response["result"].get("tools", [])

        for tool_data in tools_data:
            tool = self._convert_mcp_tool(tool_data)
            self._tools.append(tool)

    def _convert_mcp_tool(self, tool_data: dict) -> Tool:
        """Convert an MCP tool definition to a Veska Tool."""
        name = tool_data["name"]
        description = tool_data.get("description", "")
        input_schema = tool_data.get("inputSchema", {})

        # Convert MCP parameters to ToolParameters
        parameters = []
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for param_name, param_def in properties.items():
            parameters.append(ToolParameter(
                name=param_name,
                type=param_def.get("type", "string"),
                description=param_def.get("description", ""),
                required=param_name in required,
                default=param_def.get("default"),
            ))

        # Create a callable that routes to the MCP server
        server = self

        async def mcp_function(**kwargs: Any) -> str:
            result = await server.call_tool(name, kwargs)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return str(result)

        return Tool(
            name=f"{self.name}__{name}",
            description=f"[MCP:{self.name}] {description}",
            when_to_use=description,
            parameters=parameters,
            function=mcp_function,
        )

    async def _send_request(self, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC request to the MCP server."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(request) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

            line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=30
            )
            if line:
                return json.loads(line.decode())
            return None

        except (asyncio.TimeoutError, json.JSONDecodeError, OSError):
            return None

    async def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(notification) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()
        except OSError:
            pass


class MCPConnector:
    """
    Manages MCP server connections.

    Usage:
        connector = MCPConnector()

        # Add servers
        connector.add_server(MCPServer(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "..."},
        ))

        # Connect all
        await connector.connect_all()

        # Get all MCP tools (ready to register in ToolRegistry)
        tools = connector.get_all_tools()
        for tool in tools:
            registry.register(tool)

        # Cleanup
        await connector.disconnect_all()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def add_server(self, server: MCPServer) -> None:
        """Add an MCP server."""
        self._servers[server.name] = server

    def remove_server(self, name: str) -> None:
        """Remove an MCP server."""
        self._servers.pop(name, None)

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered servers. Returns success status per server."""
        results = {}
        for name, server in self._servers.items():
            results[name] = await server.connect()
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for server in self._servers.values():
            await server.disconnect()

    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get a server by name."""
        return self._servers.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Get all tools from all connected servers."""
        tools = []
        for server in self._servers.values():
            if server.connected:
                tools.extend(server.tools)
        return tools

    def get_tools(self, server_name: str) -> list[Tool]:
        """Get tools from a specific server."""
        server = self._servers.get(server_name)
        if server and server.connected:
            return server.tools
        return []

    @property
    def connected_count(self) -> int:
        return sum(1 for s in self._servers.values() if s.connected)

    @property
    def stats(self) -> dict:
        return {
            "servers": len(self._servers),
            "connected": self.connected_count,
            "total_tools": sum(len(s.tools) for s in self._servers.values() if s.connected),
        }
