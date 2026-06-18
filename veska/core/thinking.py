"""
Thinking support handler for Veska.

Manages extended thinking configuration per agent.
Handles thinking output based on user preference (discard/log/expose).
"""

from __future__ import annotations

from typing import Optional

from veska.providers.base import ThinkingConfig


class ThinkingHandler:
    """
    Handles thinking configuration and output for an agent.

    Usage:
        handler = ThinkingHandler(enabled=True, budget_tokens=10000, output="log")

        # Get config for provider
        config = handler.get_config()

        # Process thinking output from response
        handler.process(thinking_text)

        # Get stored thinking (if output is "log" or "expose")
        history = handler.get_history()
    """

    def __init__(
        self,
        enabled: bool = False,
        budget_tokens: int = 10000,
        output: str = "discard",
    ) -> None:
        self.enabled = enabled
        self.budget_tokens = budget_tokens
        self.output = output  # "discard", "log", "expose"
        self._history: list[dict] = []

    def get_config(self) -> ThinkingConfig:
        """Get ThinkingConfig for the provider."""
        return ThinkingConfig(
            enabled=self.enabled,
            budget_tokens=self.budget_tokens,
            output=self.output,
        )

    def process(self, thinking_text: Optional[str], task_id: str = "") -> Optional[str]:
        """
        Process thinking output from a response.

        Returns thinking text if output mode is "expose", else None.
        """
        if not thinking_text:
            return None

        if self.output == "discard":
            return None

        entry = {"task_id": task_id, "thinking": thinking_text}

        if self.output == "log":
            self._history.append(entry)
            return None

        if self.output == "expose":
            self._history.append(entry)
            return thinking_text

        return None

    def get_history(self) -> list[dict]:
        """Get all recorded thinking history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear thinking history."""
        self._history.clear()
