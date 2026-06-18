"""
Pre-built Project tools for Veska.

Provides: create_folder, create_project_structure, init_git, create_env_file
User just adds "project_tools" to their tools list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from veska.tools.base import Tool, ToolParameter


def _create_folder(path: str) -> str:
    """Create a folder (and parent folders if needed)."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return f"Folder created: {path}"


def _create_project_structure(root: str, structure: str) -> str:
    """
    Create a project folder structure from a description.

    structure format (one path per line):
        src/
        src/components/
        src/pages/
        src/styles/
        tests/
        config/
    """
    root_path = Path(root)
    created = []

    for line in structure.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        folder_path = root_path / line.rstrip("/")
        folder_path.mkdir(parents=True, exist_ok=True)
        created.append(str(folder_path))

    return f"Created {len(created)} folders in {root}:\n" + "\n".join(
        f"  {c}" for c in created
    )


def _init_git(directory: str) -> str:
    """Initialize a git repository."""
    try:
        result = subprocess.run(
            ["git", "init"],
            capture_output=True,
            text=True,
            cwd=directory,
        )
        if result.returncode == 0:
            # Create .gitignore
            gitignore = Path(directory) / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "node_modules/\nvenv/\n__pycache__/\n*.pyc\n.env\n.DS_Store\n"
                )
            return f"Git initialized in {directory}"
        return f"Git init failed: {result.stderr}"
    except FileNotFoundError:
        return "Error: Git not found. Install git first."


def _create_env_file(path: str, variables: str) -> str:
    """
    Create a .env file with environment variables.

    variables format (one per line):
        DATABASE_URL=postgresql://localhost:5432/mydb
        SECRET_KEY=your-secret-key
        PORT=3000
    """
    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(variables.strip() + "\n")
    return f"Environment file created: {path}"


def get_project_tools() -> list[Tool]:
    """Get all project tools."""
    return [
        Tool(
            name="create_folder",
            description="Create a folder and any necessary parent folders",
            when_to_use="When you need to create a directory structure",
            parameters=[
                ToolParameter(name="path", type="string", description="Folder path to create"),
            ],
            function=_create_folder,
        ),
        Tool(
            name="create_project_structure",
            description="Create multiple folders from a structure description",
            when_to_use="When setting up a new project and need to create the full folder structure at once",
            parameters=[
                ToolParameter(name="root", type="string", description="Root directory for the project"),
                ToolParameter(name="structure", type="string", description="Folder structure, one path per line"),
            ],
            function=_create_project_structure,
        ),
        Tool(
            name="init_git",
            description="Initialize a git repository in a directory",
            when_to_use="When starting a new project that should use git version control",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory to initialize git in"),
            ],
            function=_init_git,
        ),
        Tool(
            name="create_env_file",
            description="Create a .env file with environment variables",
            when_to_use="When you need to set up environment configuration (API keys, database URLs, ports)",
            parameters=[
                ToolParameter(name="path", type="string", description="Path for the .env file"),
                ToolParameter(name="variables", type="string", description="Environment variables, one KEY=VALUE per line"),
            ],
            function=_create_env_file,
        ),
    ]
