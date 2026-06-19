"""
Pre-built Code Runner tools for Veska.

Provides: run_python, run_node, run_command, install_package, run_tests
User just adds "code_runner" to their tools list.

When a CommandGuard is provided, all commands are checked against the
sandbox before execution. Without a guard, commands run directly.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from veska.tools.base import Tool, ToolParameter


async def _guarded_run(guard, agent_name: str, command: str, cwd: str = ".", timeout: int = 60) -> str:
    """Run a command through the CommandGuard."""
    result = await guard.run(agent_name=agent_name, command=command, cwd=cwd, timeout=timeout)
    if result.blocked:
        return f"Blocked: {result.block_reason}"
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    if result.return_code != 0:
        output += f"\nExit code: {result.return_code}"
    return output or "(no output)"


def _raw_run(command: str, cwd: str = ".", timeout: int = 60, shell: bool = True) -> str:
    """Run a command directly via subprocess (no guard)."""
    try:
        result = subprocess.run(
            command if shell else command.split(),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out ({timeout}s limit)"


def _make_run_python(guard=None, agent_name: str = "agent"):
    """Create a run_python function, optionally guarded."""

    async def run_python(code: str, cwd: str = ".") -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=cwd
        ) as f:
            f.write(code)
            f.flush()
            try:
                cmd = f"python3 {f.name}"
                if guard:
                    return await _guarded_run(guard, agent_name, cmd, cwd, timeout=30)
                return _raw_run(cmd, cwd, timeout=30)
            finally:
                Path(f.name).unlink(missing_ok=True)

    return run_python


def _make_run_node(guard=None, agent_name: str = "agent"):
    """Create a run_node function, optionally guarded."""

    async def run_node(code: str, cwd: str = ".") -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, dir=cwd
        ) as f:
            f.write(code)
            f.flush()
            try:
                cmd = f"node {f.name}"
                if guard:
                    return await _guarded_run(guard, agent_name, cmd, cwd, timeout=30)
                return _raw_run(cmd, cwd, timeout=30)
            except FileNotFoundError:
                return "Error: Node.js not found. Install Node.js first."
            finally:
                Path(f.name).unlink(missing_ok=True)

    return run_node


def _make_run_command(guard=None, agent_name: str = "agent"):
    """Create a run_command function, optionally guarded."""

    async def run_command(command: str, cwd: str = ".") -> str:
        if guard:
            return await _guarded_run(guard, agent_name, command, cwd, timeout=60)
        return _raw_run(command, cwd, timeout=60)

    return run_command


def _make_install_package(guard=None, agent_name: str = "agent"):
    """Create an install_package function, optionally guarded."""

    async def install_package(package: str, manager: str = "pip", cwd: str = ".") -> str:
        if manager == "pip":
            cmd = f"pip install {package}"
        elif manager == "npm":
            cmd = f"npm install {package}"
        else:
            return f"Error: Unknown package manager '{manager}'. Use 'pip' or 'npm'."

        if guard:
            return await _guarded_run(guard, agent_name, cmd, cwd, timeout=120)
        return _raw_run(cmd, cwd, timeout=120)

    return install_package


def _make_run_tests(guard=None, agent_name: str = "agent"):
    """Create a run_tests function, optionally guarded."""

    async def run_tests(directory: str, framework: str = "pytest") -> str:
        if framework == "pytest":
            cmd = f"python3 -m pytest {directory} -v --tb=short"
        elif framework == "jest":
            cmd = f"npx jest {directory} --verbose"
        elif framework == "unittest":
            cmd = f"python3 -m unittest discover {directory} -v"
        else:
            return f"Error: Unknown test framework '{framework}'"

        if guard:
            return await _guarded_run(guard, agent_name, cmd, directory, timeout=120)
        return _raw_run(cmd, directory, timeout=120)

    return run_tests


def get_code_runner_tools(guard=None, agent_name: str = "agent") -> list[Tool]:
    """
    Get all code runner tools.

    Args:
        guard: Optional CommandGuard for sandboxed execution.
        agent_name: Name of the agent using these tools (for security checks).
    """
    return [
        Tool(
            name="run_python",
            description="Execute Python code and return the output",
            when_to_use="When you need to run Python code to test it or generate output",
            parameters=[
                ToolParameter(name="code", type="string", description="Python code to execute"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_make_run_python(guard, agent_name),
        ),
        Tool(
            name="run_node",
            description="Execute Node.js code and return the output",
            when_to_use="When you need to run JavaScript/Node.js code to test it",
            parameters=[
                ToolParameter(name="code", type="string", description="JavaScript code to execute"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_make_run_node(guard, agent_name),
        ),
        Tool(
            name="run_command",
            description="Execute a shell command and return the output",
            when_to_use="When you need to run system commands like git, ls, or other CLI tools",
            parameters=[
                ToolParameter(name="command", type="string", description="Shell command to run"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_make_run_command(guard, agent_name),
        ),
        Tool(
            name="install_package",
            description="Install a package using pip or npm",
            when_to_use="When you need to install Python (pip) or Node.js (npm) dependencies",
            parameters=[
                ToolParameter(name="package", type="string", description="Package name to install"),
                ToolParameter(name="manager", type="string", description="Package manager: 'pip' or 'npm'", required=False, default="pip"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_make_install_package(guard, agent_name),
        ),
        Tool(
            name="run_tests",
            description="Run test suite in a directory",
            when_to_use="When you need to run tests to verify code works correctly",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory containing tests"),
                ToolParameter(name="framework", type="string", description="Test framework: 'pytest', 'jest', or 'unittest'", required=False, default="pytest"),
            ],
            function=_make_run_tests(guard, agent_name),
        ),
    ]
