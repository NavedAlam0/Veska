"""
Logging for Veska (Optional).

Optional structured logging. Developer enables this if they want
visibility into what the framework is doing.

OFF by default. User opts in.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogEntry:
    """A single log entry."""

    def __init__(
        self,
        level: LogLevel,
        source: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        self.level = level
        self.source = source
        self.message = message
        self.data = data
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "source": self.source,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def format(self) -> str:
        """Format as a readable log line."""
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        level = self.level.value.upper().ljust(7)
        data_str = f" | {json.dumps(self.data)}" if self.data else ""
        return f"[{ts}] {level} [{self.source}] {self.message}{data_str}"


class Logger:
    """
    Optional structured logger for Veska.

    OFF by default. Developer enables it in config.

    Usage:
        logger = Logger(enabled=True, level=LogLevel.DEBUG)

        # Log messages
        logger.info("orchestrator", "Plan created", {"tasks": 5})
        logger.error("backend_agent", "Tool failed", {"tool": "create_file"})

        # Add custom output (file, callback, etc.)
        logger.add_sink(my_file_sink)
        logger.add_sink(lambda entry: print(entry.format()))

        # Get logs
        logs = logger.get_logs(source="backend_agent")
    """

    def __init__(
        self,
        enabled: bool = False,
        level: LogLevel = LogLevel.INFO,
        max_entries: int = 10000,
    ) -> None:
        self._enabled = enabled
        self._level = level
        self._max_entries = max_entries
        self._entries: list[LogEntry] = []
        self._sinks: list[Callable[[LogEntry], None]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, level: Optional[LogLevel] = None) -> None:
        """Enable logging."""
        self._enabled = True
        if level:
            self._level = level

    def disable(self) -> None:
        """Disable logging."""
        self._enabled = False

    def add_sink(self, sink: Callable[[LogEntry], None]) -> None:
        """
        Add an output sink.

        A sink is any callable that receives a LogEntry.
        Could write to file, send to service, print to console, etc.
        """
        self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[LogEntry], None]) -> None:
        """Remove an output sink."""
        if sink in self._sinks:
            self._sinks.remove(sink)

    # --- Log methods ---

    def debug(self, source: str, message: str, data: Optional[dict] = None) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, source, message, data)

    def info(self, source: str, message: str, data: Optional[dict] = None) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, source, message, data)

    def warning(self, source: str, message: str, data: Optional[dict] = None) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, source, message, data)

    def error(self, source: str, message: str, data: Optional[dict] = None) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, source, message, data)

    def _log(self, level: LogLevel, source: str, message: str, data: Optional[dict] = None) -> None:
        """Internal log method."""
        if not self._enabled:
            return

        # Check level
        levels = list(LogLevel)
        if levels.index(level) < levels.index(self._level):
            return

        entry = LogEntry(level=level, source=source, message=message, data=data)

        # Store
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Send to sinks
        for sink in self._sinks:
            try:
                sink(entry)
            except Exception:
                pass

    # --- Query ---

    def get_logs(
        self,
        source: Optional[str] = None,
        level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Get log entries with optional filters."""
        entries = self._entries

        if source:
            entries = [e for e in entries if e.source == source]
        if level:
            entries = [e for e in entries if e.level == level]

        return entries[-limit:]

    def get_formatted(
        self,
        source: Optional[str] = None,
        level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> str:
        """Get formatted log output."""
        entries = self.get_logs(source=source, level=level, limit=limit)
        return "\n".join(e.format() for e in entries)

    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()

    @property
    def count(self) -> int:
        return len(self._entries)


def file_sink(path: str) -> Callable[[LogEntry], None]:
    """Create a sink that writes to a file."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def sink(entry: LogEntry) -> None:
        with open(log_path, "a") as f:
            f.write(entry.format() + "\n")

    return sink
