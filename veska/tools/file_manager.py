"""
Pre-built File Manager tools for Veska.

Provides: create_file, read_file, edit_file, delete_file, list_files, search_files
User just adds "file_manager" to their tools list. Zero code needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from veska.tools.base import Tool, ToolParameter


def _create_file(path: str, content: str) -> str:
    """Create a new file with content."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"File created: {path}"


def _read_file(path: str) -> str:
    """Read a file's content."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return file_path.read_text()


def _edit_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing text."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = file_path.read_text()
    if old_text not in content:
        raise ValueError(f"Text to replace not found in {path}")

    new_content = content.replace(old_text, new_text, 1)
    file_path.write_text(new_content)
    return f"File edited: {path}"


def _delete_file(path: str) -> str:
    """Delete a file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    file_path.unlink()
    return f"File deleted: {path}"


def _list_files(directory: str, pattern: str = "*") -> str:
    """List files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(dir_path.rglob(pattern))
    result = []
    for f in files[:100]:  # limit output
        rel = f.relative_to(dir_path)
        file_type = "DIR" if f.is_dir() else "FILE"
        result.append(f"  [{file_type}] {rel}")

    return f"Files in {directory}:\n" + "\n".join(result)


def _search_files(directory: str, query: str) -> str:
    """Search for text across files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_dir():
            continue
        try:
            content = file_path.read_text()
            for i, line in enumerate(content.split("\n"), 1):
                if query.lower() in line.lower():
                    rel = file_path.relative_to(dir_path)
                    results.append(f"  {rel}:{i}: {line.strip()}")
        except (UnicodeDecodeError, PermissionError):
            continue

        if len(results) >= 50:
            break

    if not results:
        return f"No matches found for '{query}' in {directory}"

    return f"Search results for '{query}':\n" + "\n".join(results)


def get_file_manager_tools() -> list[Tool]:
    """Get all file manager tools."""
    return [
        Tool(
            name="create_file",
            description="Create a new file with the given content",
            when_to_use="When you need to create a new code file, config file, or any text file",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to create"),
                ToolParameter(name="content", type="string", description="Content to write"),
            ],
            function=_create_file,
        ),
        Tool(
            name="read_file",
            description="Read the content of a file",
            when_to_use="When you need to see existing code or file content",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to read"),
            ],
            function=_read_file,
        ),
        Tool(
            name="edit_file",
            description="Edit a file by replacing specific text",
            when_to_use="When you need to modify existing code - replace old text with new text",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to edit"),
                ToolParameter(name="old_text", type="string", description="Text to find and replace"),
                ToolParameter(name="new_text", type="string", description="Replacement text"),
            ],
            function=_edit_file,
        ),
        Tool(
            name="delete_file",
            description="Delete a file",
            when_to_use="When you need to remove a file that is no longer needed or needs to be recreated",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to delete"),
            ],
            function=_delete_file,
        ),
        Tool(
            name="list_files",
            description="List all files and directories in a folder",
            when_to_use="When you need to see what files exist in a directory",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory path to list"),
                ToolParameter(name="pattern", type="string", description="Glob pattern to filter", required=False, default="*"),
            ],
            function=_list_files,
        ),
        Tool(
            name="search_files",
            description="Search for text across files in a directory",
            when_to_use="When you need to find where specific code, text, or patterns exist in the project",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory to search in"),
                ToolParameter(name="query", type="string", description="Text to search for"),
            ],
            function=_search_files,
        ),
    ]
