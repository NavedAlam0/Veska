"""
Pre-built Code Scanner tools for Veska.

Provides: scan_code, scan_file, scan_directory
User adds "code_scanner" to their tools list when they want optional code checks.
"""

from __future__ import annotations

from pathlib import Path

from veska.security.code_scanner import CodeScanner
from veska.tools.base import Tool, ToolParameter


DEFAULT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".php",
    ".cs",
}


def _result_to_dict(result, source: str) -> dict:
    return {
        "source": source,
        "clean": not result.has_warnings,
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
    }


def _make_scan_code(scanner: CodeScanner):
    def scan_code(code: str, language: str = "python") -> dict:
        result = scanner.scan(code, language=language)
        return _result_to_dict(result, source="<inline>")

    return scan_code


def _make_scan_file(scanner: CodeScanner):
    def scan_file(path: str, language: str = "python") -> dict:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        result = scanner.scan(file_path.read_text(), language=language)
        return _result_to_dict(result, source=str(file_path))

    return scan_file


def _make_scan_directory(scanner: CodeScanner):
    def scan_directory(
        directory: str,
        extensions: str = "",
        max_files: int = 100,
    ) -> dict:
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        selected_extensions = {
            ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
            for ext in extensions.split(",")
            if ext.strip()
        } or DEFAULT_EXTENSIONS

        files_scanned = 0
        warnings = []
        for file_path in sorted(dir_path.rglob("*")):
            if files_scanned >= max_files:
                break
            if not file_path.is_file() or file_path.suffix not in selected_extensions:
                continue

            try:
                result = scanner.scan(file_path.read_text(), language=file_path.suffix.lstrip("."))
            except UnicodeDecodeError:
                continue

            files_scanned += 1
            for warning in result.warnings:
                warnings.append({"file": str(file_path), **warning})

        return {
            "source": str(dir_path),
            "clean": len(warnings) == 0,
            "files_scanned": files_scanned,
            "warning_count": len(warnings),
            "warnings": warnings,
        }

    return scan_directory


def get_code_scanner_tools(scanner: CodeScanner | None = None) -> list[Tool]:
    """Get code scanner tools."""
    scanner = scanner or CodeScanner()
    return [
        Tool(
            name="scan_code",
            description="Scan inline source code for common security issues",
            when_to_use="When you need to check code content for obvious security risks",
            parameters=[
                ToolParameter(name="code", type="string", description="Source code to scan"),
                ToolParameter(name="language", type="string", description="Programming language", required=False, default="python"),
            ],
            function=_make_scan_code(scanner),
        ),
        Tool(
            name="scan_file",
            description="Scan a source file for common security issues",
            when_to_use="When you need to check one generated or existing source file",
            parameters=[
                ToolParameter(name="path", type="string", description="File path to scan"),
                ToolParameter(name="language", type="string", description="Programming language", required=False, default="python"),
            ],
            function=_make_scan_file(scanner),
        ),
        Tool(
            name="scan_directory",
            description="Scan source files in a directory for common security issues",
            when_to_use="When you need to check a generated project or folder",
            parameters=[
                ToolParameter(name="directory", type="string", description="Directory path to scan"),
                ToolParameter(name="extensions", type="string", description="Comma-separated extensions, for example .py,.js", required=False, default=""),
                ToolParameter(name="max_files", type="integer", description="Maximum number of files to scan", required=False, default=100),
            ],
            function=_make_scan_directory(scanner),
        ),
    ]
