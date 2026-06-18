"""
Pre-built Code Runner tools for Veska.

Provides: run_python, run_node, run_command, install_package, run_tests
User just adds "code_runner" to their tools list.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from veska.tools.base import Tool, ToolParameter


def _run_python(code: str, cwd: str = ".") -> str:
    """Run Python code and return output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=cwd
    ) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30s limit)"
        finally:
            Path(f.name).unlink(missing_ok=True)


def _run_node(code: str, cwd: str = ".") -> str:
    """Run Node.js code and return output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, dir=cwd
    ) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["node", f.name],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30s limit)"
        except FileNotFoundError:
            return "Error: Node.js not found. Install Node.js first."
        finally:
            Path(f.name).unlink(missing_ok=True)


def _run_command(command: str, cwd: str = ".") -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s limit)"


def _install_package(package: str, manager: str = "pip", cwd: str = ".") -> str:
    """Install a package using pip or npm."""
    if manager == "pip":
        cmd = f"pip install {package}"
    elif manager == "npm":
        cmd = f"npm install {package}"
    else:
        return f"Error: Unknown package manager '{manager}'. Use 'pip' or 'npm'."

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        )
        if result.returncode == 0:
            return f"Successfully installed {package}"
        return f"Failed to install {package}:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: Installation of {package} timed out"


def _run_tests(directory: str, framework: str = "pytest") -> str:
    """Run tests in a directory."""
    if framework == "pytest":
        cmd = f"python3 -m pytest {directory} -v --tb=short"
    elif framework == "jest":
        cmd = f"npx jest {directory} --verbose"
    elif framework == "unittest":
        cmd = f"python3 -m unittest discover {directory} -v"
    else:
        return f"Error: Unknown test framework '{framework}'"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=directory,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Tests timed out (120s limit)"


def get_code_runner_tools() -> list[Tool]:
    """Get all code runner tools."""
    return [
        Tool(
            name="run_python",
            description="Execute Python code and return the output",
            when_to_use="When you need to run Python code to test it or generate output",
            parameters=[
                ToolParameter(name="code", type="string", description="Python code to execute"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_run_python,
        ),
        Tool(
            name="run_node",
            description="Execute Node.js code and return the output",
            when_to_use="When you need to run JavaScript/Node.js code to test it",
            parameters=[
                ToolParameter(name="code", type="string", description="JavaScript code to execute"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_run_node,
        ),
        Tool(
            name="run_command",
            description="Execute a shell command and return the output",
            when_to_use="When you need to run system commands like git, ls, or other CLI tools",
            parameters=[
                ToolParameter(name="command", type="string", description="Shell command to run"),
                ToolParameter(name="cwd", type="string", description="Working directory", required=False, default="."),
            ],
            function=_run_command,
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
            function=_install_package,
        ),
        Tool(
            name="run_tests",
            description="Run test suite in a directory",
            when_to_use="When you need to run tests to verify code works correctly",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory containing tests"),
                ToolParameter(name="framework", type="string", description="Test framework: 'pytest', 'jest', or 'unittest'", required=False, default="pytest"),
            ],
            function=_run_tests,
        ),
    ]
