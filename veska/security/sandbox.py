"""
Security & Sandboxing for Veska.

Three zones:
  Zone 1: Framework code - FULLY LOCKED (no agent can touch)
  Zone 2: Project output - full freedom inside own territory, read-only for others
  Zone 3: User's system - FULLY BLOCKED

Each agent has its own territory (folder). Full freedom inside.
Can't edit other agents' folders. Can't touch framework or system files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


# Commands that are always blocked regardless of context
BLOCKED_COMMANDS = {
    "sudo",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "init",
    "systemctl",
    "mkfs",
    "dd",
    "fdisk",
    "mount",
    "umount",
    "passwd",
    "useradd",
    "userdel",
    "groupadd",
    "chown",
    "chmod",  # system-level only, allowed inside project
}

# Patterns in commands that are always dangerous
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "> /dev/",
    ":(){ :|:& };:",  # fork bomb
    "mv / ",
]

# Regex patterns for more complex dangerous commands (e.g., curl <url> | bash)
BLOCKED_REGEX_PATTERNS = [
    re.compile(r"(wget|curl)\s+.*\|\s*(sh|bash|zsh)"),
]


class SecurityZone:
    """Defines what an agent can access."""

    def __init__(
        self,
        agent_name: str,
        territory: str,
        project_root: str,
        framework_root: str,
        read_access: Optional[list[str]] = None,
    ) -> None:
        self.agent_name = agent_name
        self.territory = Path(territory).resolve()
        self.project_root = Path(project_root).resolve()
        self.framework_root = Path(framework_root).resolve()
        # Additional folders this agent can READ (not write)
        self.read_access = [Path(p).resolve() for p in (read_access or [])]


class Sandbox:
    """
    Enforces security boundaries for all agents.

    Usage:
        sandbox = Sandbox(
            project_root="/projects/blog-app",
            framework_root="/path/to/veska/agents",
        )

        # Set agent territories
        sandbox.set_territory("backend_agent", "/projects/blog-app/backend")
        sandbox.set_territory("frontend_agent", "/projects/blog-app/frontend")

        # Check if action is allowed
        sandbox.check_path_access("backend_agent", "/projects/blog-app/backend/server.py", "write")  # OK
        sandbox.check_path_access("backend_agent", "/projects/blog-app/frontend/App.jsx", "write")  # BLOCKED
        sandbox.check_path_access("backend_agent", "/etc/passwd", "read")  # BLOCKED

        # Check if command is safe
        sandbox.check_command("backend_agent", "python server.py")  # OK
        sandbox.check_command("backend_agent", "sudo rm -rf /")  # BLOCKED
    """

    def __init__(
        self,
        project_root: str,
        framework_root: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.framework_root = (
            Path(framework_root).resolve()
            if framework_root
            else Path(__file__).parent.parent.resolve()
        )
        self._zones: dict[str, SecurityZone] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        """Disable sandbox (for testing only)."""
        self._enabled = False

    def enable(self) -> None:
        """Enable sandbox."""
        self._enabled = True

    # --- Territory management ---

    def set_territory(
        self,
        agent_name: str,
        territory: str,
        read_access: Optional[list[str]] = None,
    ) -> None:
        """
        Set an agent's territory (folder it has full access to).

        Args:
            agent_name: Agent's name.
            territory: Absolute path to agent's folder.
            read_access: Additional folders agent can read (not write).
        """
        self._zones[agent_name] = SecurityZone(
            agent_name=agent_name,
            territory=territory,
            project_root=str(self.project_root),
            framework_root=str(self.framework_root),
            read_access=read_access,
        )

    def get_territory(self, agent_name: str) -> Optional[Path]:
        """Get an agent's territory path."""
        zone = self._zones.get(agent_name)
        return zone.territory if zone else None

    # --- Path access checks ---

    def check_path_access(
        self, agent_name: str, path: str, mode: str = "read"
    ) -> SecurityResult:
        """
        Check if an agent can access a path.

        Args:
            agent_name: Agent's name.
            path: File/folder path to access.
            mode: "read" or "write".

        Returns:
            SecurityResult with allowed/blocked status.
        """
        if not self._enabled:
            return SecurityResult(allowed=True)

        resolved = Path(path).resolve()

        # Zone 1: Framework code - FULLY LOCKED
        if self._is_framework_path(resolved):
            return SecurityResult(
                allowed=False,
                reason=f"Framework code is locked. Agent '{agent_name}' cannot access {path}",
                zone="framework",
            )

        # Zone 3: Outside project - FULLY BLOCKED
        if not self._is_project_path(resolved):
            return SecurityResult(
                allowed=False,
                reason=f"Path '{path}' is outside the project. Agent '{agent_name}' has no access.",
                zone="system",
            )

        # Zone 2: Inside project
        zone = self._zones.get(agent_name)
        if not zone:
            # Agent has no territory defined - allow read, block write inside project
            if mode == "read":
                return SecurityResult(allowed=True)
            return SecurityResult(
                allowed=False,
                reason=f"Agent '{agent_name}' has no territory defined.",
                zone="project",
            )

        # Write access - must be inside own territory
        if mode == "write":
            if self._is_in_territory(resolved, zone.territory):
                return SecurityResult(allowed=True)

            # Check if it's a shared project-root file (like .env, README)
            if resolved.parent == self.project_root:
                return SecurityResult(
                    allowed=False,
                    reason=(
                        f"Shared file '{path}' must be modified through the File Manager. "
                        f"Agent '{agent_name}' cannot write directly."
                    ),
                    zone="shared",
                )

            return SecurityResult(
                allowed=False,
                reason=(
                    f"Agent '{agent_name}' can only write inside '{zone.territory}'. "
                    f"Path '{path}' is outside its territory."
                ),
                zone="other_agent",
            )

        # Read access - own territory + read_access folders + project root files
        if mode == "read":
            # Can always read own territory
            if self._is_in_territory(resolved, zone.territory):
                return SecurityResult(allowed=True)

            # Can read project root files
            if resolved.parent == self.project_root:
                return SecurityResult(allowed=True)

            # Can read explicit read_access folders
            for read_path in zone.read_access:
                if self._is_in_territory(resolved, read_path):
                    return SecurityResult(allowed=True)

            # Can read other areas inside project (read-only through file manager)
            if self._is_project_path(resolved):
                return SecurityResult(allowed=True)

        return SecurityResult(allowed=True)

    # --- Command checks ---

    def check_command(self, agent_name: str, command: str) -> SecurityResult:
        """
        Check if a command is safe to execute.

        Args:
            agent_name: Agent's name.
            command: Shell command to check.

        Returns:
            SecurityResult with allowed/blocked status.
        """
        if not self._enabled:
            return SecurityResult(allowed=True)

        command_lower = command.lower().strip()

        # Check blocked patterns (exact substring)
        for pattern in BLOCKED_PATTERNS:
            if pattern in command_lower:
                return SecurityResult(
                    allowed=False,
                    reason=f"Dangerous command pattern detected: '{pattern}'",
                    zone="command",
                )

        # Check blocked regex patterns
        for regex in BLOCKED_REGEX_PATTERNS:
            match = regex.search(command_lower)
            if match:
                return SecurityResult(
                    allowed=False,
                    reason=f"Dangerous command pattern detected: '{match.group()}'",
                    zone="command",
                )

        # Check if command starts with blocked command
        first_word = command_lower.split()[0] if command_lower else ""

        # Strip path prefixes (e.g., /usr/bin/sudo -> sudo)
        first_word = first_word.split("/")[-1]

        if first_word in BLOCKED_COMMANDS:
            # chmod is allowed inside project territory
            if first_word == "chmod":
                zone = self._zones.get(agent_name)
                if zone and self._command_targets_territory(command, zone.territory):
                    return SecurityResult(allowed=True)

            return SecurityResult(
                allowed=False,
                reason=f"Command '{first_word}' is blocked for security.",
                zone="command",
            )

        return SecurityResult(allowed=True)

    # --- Internal helpers ---

    def _is_framework_path(self, path: Path) -> bool:
        """Check if path is inside the framework code."""
        try:
            path.relative_to(self.framework_root)
            return True
        except ValueError:
            return False

    def _is_project_path(self, path: Path) -> bool:
        """Check if path is inside the project."""
        try:
            path.relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def _is_in_territory(self, path: Path, territory: Path) -> bool:
        """Check if path is inside a territory."""
        try:
            path.relative_to(territory)
            return True
        except ValueError:
            return False

    def _command_targets_territory(self, command: str, territory: Path) -> bool:
        """Check if a command's file arguments are inside a territory."""
        parts = command.split()
        for part in parts[1:]:
            if part.startswith("-"):
                continue
            try:
                resolved = Path(part).resolve()
                if not self._is_in_territory(resolved, territory):
                    return False
            except (OSError, ValueError):
                continue
        return True


class SecurityResult:
    """Result of a security check."""

    def __init__(
        self,
        allowed: bool,
        reason: str = "",
        zone: str = "",
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.zone = zone  # framework, system, project, shared, other_agent, command

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        status = "ALLOWED" if self.allowed else "BLOCKED"
        return f"SecurityResult({status}: {self.reason})"
