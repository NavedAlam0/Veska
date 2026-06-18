"""
Command Guard for Veska.

Wraps command execution with security checks.
Agents use this instead of running commands directly.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Optional

from veska.security.sandbox import Sandbox, SecurityResult


class CommandResult:
    """Result from running a command."""

    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        return_code: int = 0,
        blocked: bool = False,
        block_reason: str = "",
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.blocked = blocked
        self.block_reason = block_reason


class CommandGuard:
    """
    Secure command execution for agents.

    All commands go through security checks before execution.
    Commands run inside the agent's territory (working directory).

    Usage:
        guard = CommandGuard(sandbox)

        result = await guard.run(
            agent_name="backend_agent",
            command="python server.py",
            cwd="/projects/blog-app/backend",
        )
    """

    def __init__(
        self,
        sandbox: Sandbox,
        timeout: int = 60,
    ) -> None:
        self.sandbox = sandbox
        self.timeout = timeout

    async def run(
        self,
        agent_name: str,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """
        Run a command with security checks.

        Args:
            agent_name: Agent requesting the command.
            command: Shell command to run.
            cwd: Working directory (defaults to agent's territory).
            timeout: Command timeout in seconds.
        """
        # Security check
        check = self.sandbox.check_command(agent_name, command)
        if not check.allowed:
            return CommandResult(
                success=False,
                blocked=True,
                block_reason=check.reason,
            )

        # Determine working directory
        if cwd is None:
            territory = self.sandbox.get_territory(agent_name)
            cwd = str(territory) if territory else None

        # Execute command
        run_timeout = timeout or self.timeout
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=run_timeout,
            )

            return CommandResult(
                success=process.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=process.returncode or 0,
            )

        except asyncio.TimeoutError:
            process.kill()
            return CommandResult(
                success=False,
                stderr=f"Command timed out after {run_timeout}s",
                return_code=-1,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                stderr=str(e),
                return_code=-1,
            )
