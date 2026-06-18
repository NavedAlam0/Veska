"""
Error Recovery System for Veska.

Three severity levels:
  Level 1: Simple retry - agent retries the task (auto-handled)
  Level 2: Agent-level fix - agent gets error context and tries a different approach
  Level 3: Discussion room - multiple agents discuss the problem and propose a fix

Fix Reports coordinate cross-agent fixes when one agent's error
requires changes in another agent's territory.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field


class ErrorSeverity(int, Enum):
    """Error severity levels."""

    LEVEL_1 = 1  # Simple retry
    LEVEL_2 = 2  # Agent-level fix with error context
    LEVEL_3 = 3  # Discussion room needed


class ErrorCategory(str, Enum):
    """Categories of errors."""

    TOOL_FAILURE = "tool_failure"
    CODE_ERROR = "code_error"
    DEPENDENCY_ERROR = "dependency_error"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    PERMISSION_ERROR = "permission_error"
    CROSS_AGENT = "cross_agent"
    UNKNOWN = "unknown"


class DetectedError(BaseModel):
    """An error detected by the system."""

    id: str = Field(default_factory=lambda: f"err_{uuid.uuid4().hex[:8]}")
    agent_name: str
    task_id: str = ""
    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.LEVEL_1
    details: Optional[dict] = None
    timestamp: float = Field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None


class FixReport(BaseModel):
    """
    Coordinates a cross-agent fix.

    When one agent's error requires changes in another agent's territory,
    a FixReport is created and sent through the message bus.
    """

    id: str = Field(default_factory=lambda: f"fix_{uuid.uuid4().hex[:8]}")
    error_id: str
    from_agent: str  # agent that found the problem
    to_agent: str  # agent that needs to fix it
    problem: str  # what went wrong
    suggestion: str = ""  # suggested fix
    affected_files: list[str] = Field(default_factory=list)
    status: str = "pending"  # pending, accepted, rejected, completed
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[str] = None


class DiscussionMessage(BaseModel):
    """A single message in a discussion room."""

    agent_name: str
    content: str
    timestamp: float = Field(default_factory=time.time)


class DiscussionResult(BaseModel):
    """Result from a discussion room session."""

    error_id: str
    participants: list[str]
    messages: list[DiscussionMessage] = Field(default_factory=list)
    conclusion: str = ""
    fix_reports: list[FixReport] = Field(default_factory=list)
    resolved: bool = False


# --- Error Detector ---


# Patterns that indicate specific error categories
CATEGORY_PATTERNS = {
    ErrorCategory.TOOL_FAILURE: [
        "tool failed", "tool error", "unknown tool",
        "tool not found", "execution failed",
    ],
    ErrorCategory.CODE_ERROR: [
        "syntaxerror", "nameerror", "typeerror", "valueerror",
        "importerror", "attributeerror", "indexerror", "keyerror",
        "indentationerror", "runtime error", "compilation error",
    ],
    ErrorCategory.DEPENDENCY_ERROR: [
        "module not found", "no module named", "package not found",
        "import error", "dependency", "not installed",
    ],
    ErrorCategory.TIMEOUT: [
        "timeout", "timed out", "deadline exceeded",
    ],
    ErrorCategory.PROVIDER_ERROR: [
        "api error", "rate limit", "token limit", "context length",
        "model error", "provider error", "authentication",
    ],
    ErrorCategory.PERMISSION_ERROR: [
        "permission denied", "access denied", "forbidden",
        "not allowed", "blocked", "security",
    ],
    ErrorCategory.CROSS_AGENT: [
        "depends on", "waiting for", "needs output from",
        "incompatible", "interface mismatch", "contract violation",
    ],
}

# Categories that typically require higher severity
SEVERITY_MAP = {
    ErrorCategory.TOOL_FAILURE: ErrorSeverity.LEVEL_1,
    ErrorCategory.CODE_ERROR: ErrorSeverity.LEVEL_2,
    ErrorCategory.DEPENDENCY_ERROR: ErrorSeverity.LEVEL_1,
    ErrorCategory.TIMEOUT: ErrorSeverity.LEVEL_1,
    ErrorCategory.PROVIDER_ERROR: ErrorSeverity.LEVEL_1,
    ErrorCategory.PERMISSION_ERROR: ErrorSeverity.LEVEL_2,
    ErrorCategory.CROSS_AGENT: ErrorSeverity.LEVEL_3,
    ErrorCategory.UNKNOWN: ErrorSeverity.LEVEL_1,
}


class ErrorDetector:
    """
    Detects, classifies, and tracks errors.

    Usage:
        detector = ErrorDetector()

        # Detect an error
        error = detector.detect(
            agent_name="backend_agent",
            task_id="task_1",
            message="SyntaxError: invalid syntax in server.py line 42",
        )
        # error.severity == LEVEL_2 (code error)
        # error.category == CODE_ERROR

        # Check if retry count exceeded (escalates severity)
        error = detector.detect(
            agent_name="backend_agent",
            task_id="task_1",
            message="Same error again",
            retry_count=3,
        )
        # error.severity == LEVEL_3 (escalated after multiple retries)
    """

    def __init__(self) -> None:
        self._errors: list[DetectedError] = []
        self._error_counts: dict[str, int] = {}  # task_id -> count

    def detect(
        self,
        agent_name: str,
        message: str,
        task_id: str = "",
        retry_count: int = 0,
        details: Optional[dict] = None,
    ) -> DetectedError:
        """
        Detect and classify an error.

        Args:
            agent_name: Which agent hit the error.
            message: The error message.
            task_id: Which task was running.
            retry_count: How many times this task has been retried.
            details: Additional error context.

        Returns:
            DetectedError with category and severity.
        """
        category = self._classify_category(message)
        severity = self._classify_severity(category, retry_count, message)

        error = DetectedError(
            agent_name=agent_name,
            task_id=task_id,
            message=message,
            category=category,
            severity=severity,
            details=details,
        )

        self._errors.append(error)

        # Track error count per task
        if task_id:
            self._error_counts[task_id] = self._error_counts.get(task_id, 0) + 1

        return error

    def resolve(self, error_id: str, resolution: str = "") -> None:
        """Mark an error as resolved."""
        for error in self._errors:
            if error.id == error_id:
                error.resolved = True
                error.resolution = resolution
                break

    def get_unresolved(self, agent_name: Optional[str] = None) -> list[DetectedError]:
        """Get all unresolved errors, optionally filtered by agent."""
        errors = [e for e in self._errors if not e.resolved]
        if agent_name:
            errors = [e for e in errors if e.agent_name == agent_name]
        return errors

    def get_error_count(self, task_id: str) -> int:
        """Get how many errors a task has encountered."""
        return self._error_counts.get(task_id, 0)

    def get_history(self, limit: int = 50) -> list[DetectedError]:
        """Get recent error history."""
        return self._errors[-limit:]

    def _classify_category(self, message: str) -> ErrorCategory:
        """Classify an error message into a category."""
        message_lower = message.lower()

        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return category

        return ErrorCategory.UNKNOWN

    def _classify_severity(
        self, category: ErrorCategory, retry_count: int, message: str
    ) -> ErrorSeverity:
        """
        Determine error severity.

        Escalation rules:
          - Base severity from category
          - 2+ retries on same task -> escalate to LEVEL_2
          - 3+ retries on same task -> escalate to LEVEL_3
          - Cross-agent errors -> always LEVEL_3
        """
        base_severity = SEVERITY_MAP.get(category, ErrorSeverity.LEVEL_1)

        # Escalate based on retry count
        if retry_count >= 3:
            return ErrorSeverity.LEVEL_3
        if retry_count >= 2 and base_severity == ErrorSeverity.LEVEL_1:
            return ErrorSeverity.LEVEL_2

        return base_severity

    @property
    def stats(self) -> dict:
        """Get error statistics."""
        total = len(self._errors)
        resolved = sum(1 for e in self._errors if e.resolved)
        by_severity = {}
        for s in ErrorSeverity:
            by_severity[s.name] = sum(1 for e in self._errors if e.severity == s)
        by_category = {}
        for c in ErrorCategory:
            count = sum(1 for e in self._errors if e.category == c)
            if count > 0:
                by_category[c.value] = count

        return {
            "total": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "by_severity": by_severity,
            "by_category": by_category,
        }


# --- Discussion Room ---


class DiscussionRoom:
    """
    Level 3 error resolution: agents discuss the problem.

    When a complex error needs multiple agents to understand and fix,
    they enter a discussion room. Each agent provides its perspective,
    and together they produce fix reports.

    Usage:
        room = DiscussionRoom(
            error=detected_error,
            participants=["backend_agent", "frontend_agent", "architect"],
        )

        # Each agent contributes (called by orchestrator)
        await room.contribute("backend_agent", agent_provider, agent_tools)
        await room.contribute("frontend_agent", agent_provider, agent_tools)

        # Get the result
        result = room.conclude()
        # result.fix_reports -> list of FixReports for each agent
    """

    def __init__(
        self,
        error: DetectedError,
        participants: list[str],
    ) -> None:
        self.error = error
        self.participants = participants
        self._messages: list[DiscussionMessage] = []
        self._fix_reports: list[FixReport] = []
        self._concluded = False

    def add_message(self, agent_name: str, content: str) -> None:
        """Add a discussion message from an agent."""
        if agent_name not in self.participants:
            return

        self._messages.append(DiscussionMessage(
            agent_name=agent_name,
            content=content,
        ))

    async def contribute(
        self,
        agent_name: str,
        provider: Any,
        context: str = "",
    ) -> str:
        """
        Have an agent contribute to the discussion using AI.

        Args:
            agent_name: The contributing agent.
            provider: The agent's AI provider.
            context: Additional context for the agent.

        Returns:
            The agent's contribution.
        """
        if not provider:
            return ""

        # Build the discussion prompt
        prompt = self._build_discussion_prompt(agent_name, context)

        from veska.providers.base import Message

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=self._format_discussion_so_far()),
        ]

        response = await provider.chat(messages=messages)
        contribution = response.content

        self.add_message(agent_name, contribution)
        return contribution

    def propose_fix(
        self,
        from_agent: str,
        to_agent: str,
        problem: str,
        suggestion: str = "",
        affected_files: Optional[list[str]] = None,
    ) -> FixReport:
        """Propose a fix that another agent needs to implement."""
        report = FixReport(
            error_id=self.error.id,
            from_agent=from_agent,
            to_agent=to_agent,
            problem=problem,
            suggestion=suggestion,
            affected_files=affected_files or [],
        )
        self._fix_reports.append(report)
        return report

    def conclude(self, conclusion: str = "") -> DiscussionResult:
        """
        Conclude the discussion and produce results.

        Args:
            conclusion: Summary of what was decided.

        Returns:
            DiscussionResult with messages and fix reports.
        """
        self._concluded = True

        if not conclusion and self._messages:
            # Auto-generate conclusion from last message
            conclusion = f"Discussion concluded with {len(self._messages)} messages and {len(self._fix_reports)} fix reports."

        return DiscussionResult(
            error_id=self.error.id,
            participants=self.participants,
            messages=self._messages,
            conclusion=conclusion,
            fix_reports=self._fix_reports,
            resolved=len(self._fix_reports) > 0 or bool(conclusion),
        )

    def _build_discussion_prompt(self, agent_name: str, context: str) -> str:
        """Build the system prompt for a discussion participant."""
        return f"""You are agent '{agent_name}' in a multi-agent discussion room.

An error occurred that requires collaboration to resolve.

ERROR:
  Agent: {self.error.agent_name}
  Category: {self.error.category.value}
  Message: {self.error.message}
  Task: {self.error.task_id}

{f"ADDITIONAL CONTEXT:{chr(10)}{context}" if context else ""}

Your job:
1. Analyze the error from your perspective
2. Explain what you think caused it
3. Suggest what needs to change and who should change it
4. If another agent needs to make changes, clearly state:
   - Which agent
   - What files/code need to change
   - What the fix should be

Be specific and actionable. Keep it concise."""

    def _format_discussion_so_far(self) -> str:
        """Format the discussion history for the next participant."""
        if not self._messages:
            return "You are the first to speak. Analyze the error and share your perspective."

        parts = ["Discussion so far:\n"]
        for msg in self._messages:
            parts.append(f"[{msg.agent_name}]: {msg.content}\n")
        parts.append("\nNow share your perspective. Agree, disagree, or add new insights.")
        return "\n".join(parts)

    @property
    def is_concluded(self) -> bool:
        return self._concluded

    @property
    def message_count(self) -> int:
        return len(self._messages)


# --- Fix Coordinator ---


class FixCoordinator:
    """
    Coordinates cross-agent fixes.

    When a discussion room produces fix reports, the coordinator
    ensures each fix is delivered to the right agent and tracked
    to completion.

    Usage:
        coordinator = FixCoordinator()

        # Add fix reports from discussion
        coordinator.add_reports(discussion_result.fix_reports)

        # Get pending fixes for an agent
        fixes = coordinator.get_pending("backend_agent")

        # Agent completes a fix
        coordinator.complete("fix_abc123", result="Fixed the API endpoint")

        # Check if all fixes for an error are done
        coordinator.is_error_resolved("err_abc123")
    """

    def __init__(self) -> None:
        self._reports: dict[str, FixReport] = {}

    def add_report(self, report: FixReport) -> None:
        """Add a fix report to track."""
        self._reports[report.id] = report

    def add_reports(self, reports: list[FixReport]) -> None:
        """Add multiple fix reports."""
        for report in reports:
            self.add_report(report)

    def get_pending(self, agent_name: str) -> list[FixReport]:
        """Get all pending fixes for an agent."""
        return [
            r for r in self._reports.values()
            if r.to_agent == agent_name and r.status == "pending"
        ]

    def get_report(self, report_id: str) -> Optional[FixReport]:
        """Get a fix report by ID."""
        return self._reports.get(report_id)

    def accept(self, report_id: str) -> None:
        """Mark a fix report as accepted (agent will work on it)."""
        report = self._reports.get(report_id)
        if report:
            report.status = "accepted"

    def reject(self, report_id: str, reason: str = "") -> None:
        """Mark a fix report as rejected."""
        report = self._reports.get(report_id)
        if report:
            report.status = "rejected"
            report.result = reason

    def complete(self, report_id: str, result: str = "") -> None:
        """Mark a fix report as completed."""
        report = self._reports.get(report_id)
        if report:
            report.status = "completed"
            report.result = result
            report.completed_at = time.time()

    def is_error_resolved(self, error_id: str) -> bool:
        """Check if all fix reports for an error are resolved."""
        reports = [
            r for r in self._reports.values()
            if r.error_id == error_id
        ]
        if not reports:
            return False
        return all(r.status in ("completed", "rejected") for r in reports)

    def get_all(self, status: Optional[str] = None) -> list[FixReport]:
        """Get all fix reports, optionally filtered by status."""
        reports = list(self._reports.values())
        if status:
            reports = [r for r in reports if r.status == status]
        return reports

    @property
    def stats(self) -> dict:
        """Get fix coordination statistics."""
        total = len(self._reports)
        by_status = {}
        for status in ("pending", "accepted", "rejected", "completed"):
            count = sum(1 for r in self._reports.values() if r.status == status)
            if count > 0:
                by_status[status] = count

        return {
            "total": total,
            "by_status": by_status,
        }
