"""
Code Scanner for Veska (Optional).

Scans generated code for common security issues before saving.
Developer enables this if they want extra security checks.
"""

from __future__ import annotations

import re
from typing import Optional


class ScanResult:
    """Result from scanning code."""

    def __init__(self) -> None:
        self.warnings: list[dict] = []

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_warning(
        self, severity: str, message: str, line: Optional[int] = None
    ) -> None:
        self.warnings.append({
            "severity": severity,
            "message": message,
            "line": line,
        })

    def __repr__(self) -> str:
        if not self.has_warnings:
            return "ScanResult(clean)"
        return f"ScanResult({len(self.warnings)} warnings)"


# Patterns to scan for
SCAN_PATTERNS = [
    {
        "name": "hardcoded_password",
        "pattern": r'(?:password|passwd|pwd|secret)\s*=\s*["\'][^"\']+["\']',
        "severity": "high",
        "message": "Possible hardcoded password detected",
    },
    {
        "name": "sql_injection",
        "pattern": r'(?:execute|cursor\.execute)\s*\(\s*["\'].*%s.*["\']|(?:execute|cursor\.execute)\s*\(\s*f["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*\{',
        "severity": "high",
        "message": "Possible SQL injection - use parameterized queries",
    },
    {
        "name": "eval_usage",
        "pattern": r'\beval\s*\(|\bexec\s*\(',
        "severity": "high",
        "message": "eval/exec usage detected - potential code injection risk",
    },
    {
        "name": "hardcoded_api_key",
        "pattern": r'(?:api_key|apikey|api_secret|token)\s*=\s*["\'][a-zA-Z0-9_\-]{20,}["\']',
        "severity": "high",
        "message": "Possible hardcoded API key detected - use environment variables",
    },
    {
        "name": "http_insecure",
        "pattern": r'http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)',
        "severity": "low",
        "message": "Insecure HTTP URL detected - consider using HTTPS",
    },
    {
        "name": "debug_enabled",
        "pattern": r'(?:DEBUG|debug)\s*=\s*True',
        "severity": "low",
        "message": "Debug mode enabled - ensure this is disabled in production",
    },
]


class CodeScanner:
    """
    Scans code for security issues.

    Optional - developer enables this if they want extra checks.

    Usage:
        scanner = CodeScanner()
        result = scanner.scan(code, language="python")

        if result.has_warnings:
            for w in result.warnings:
                print(f"[{w['severity']}] {w['message']}")
    """

    def __init__(self, extra_patterns: Optional[list[dict]] = None) -> None:
        self.patterns = list(SCAN_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def scan(self, code: str, language: str = "python") -> ScanResult:
        """
        Scan code for security issues.

        Args:
            code: Source code to scan.
            language: Programming language (for language-specific checks).

        Returns:
            ScanResult with any warnings found.
        """
        result = ScanResult()

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            for pattern_def in self.patterns:
                if re.search(pattern_def["pattern"], line, re.IGNORECASE):
                    result.add_warning(
                        severity=pattern_def["severity"],
                        message=f"{pattern_def['message']} (line {i})",
                        line=i,
                    )

        return result
