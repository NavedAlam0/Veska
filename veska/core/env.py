"""
Environment variable loader for Veska.

Auto-loads .env file at startup. Searches current directory
and walks up parent directories (like git finds .git/).

Developer puts keys in .env, Veska picks them up automatically.
No import os, no load_dotenv(), nothing extra.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_loaded = False


def load_env(start_dir: Optional[str] = None) -> bool:
    """
    Auto-load .env file.

    Searches from start_dir (or cwd) upward until it finds a .env file.
    Loads all KEY=VALUE pairs into os.environ (won't overwrite existing vars).

    Returns True if a .env file was found and loaded.
    """
    global _loaded

    start = Path(start_dir) if start_dir else Path.cwd()
    env_file = _find_env_file(start)

    if env_file is None:
        return False

    _parse_env_file(env_file)
    _loaded = True
    return True


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable by name.

    Auto-loads .env on first call if not already loaded.
    """
    if not _loaded:
        load_env()

    return os.environ.get(name, default)


def require_env(name: str) -> str:
    """
    Get a required environment variable.

    Raises an error with a helpful message if not found.
    """
    value = get_env(name)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{name}' is required but not set. "
            f"Add it to your .env file or set it in your environment."
        )
    return value


def resolve_env_vars(var_names: list[str]) -> dict[str, str]:
    """
    Resolve a list of environment variable names to their values.

    Used by MCPServer — user passes ["GITHUB_TOKEN", "NOTION_TOKEN"],
    this returns {"GITHUB_TOKEN": "actual-value", ...}.

    Skips variables that are not set (no error).
    """
    if not _loaded:
        load_env()

    result = {}
    for name in var_names:
        value = os.environ.get(name)
        if value is not None:
            result[name] = value
    return result


def _find_env_file(start: Path) -> Optional[Path]:
    """Search for .env file starting from start directory, walking up."""
    current = start.resolve()

    while True:
        env_path = current / ".env"
        if env_path.is_file():
            return env_path

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent

    return None


def _parse_env_file(path: Path) -> None:
    """
    Parse a .env file and load into os.environ.

    Supports:
      KEY=value
      KEY="value with spaces"
      KEY='value with spaces'
      # comments
      empty lines

    Does NOT overwrite existing environment variables.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    for line in text.splitlines():
        line = line.strip()

        # Skip empty lines, comments, and lines with null bytes
        if not line or line.startswith("#") or "\x00" in line:
            continue

        # Split on first =
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Remove surrounding quotes
        if len(value) >= 2:
            if (value[0] == '"' and value[-1] == '"') or \
               (value[0] == "'" and value[-1] == "'"):
                value = value[1:-1]

        # Don't overwrite existing env vars
        if key not in os.environ:
            os.environ[key] = value
