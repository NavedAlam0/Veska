"""
Base provider interface.

All AI model providers (Claude, OpenAI) implement this.
Agents don't know which provider they're using.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking."""

    enabled: bool = False
    budget_tokens: int = 10000
    output: str = "discard"  # "discard", "log", "expose"


class ProviderResponse(BaseModel):
    """Standardized response from any provider."""

    content: str = ""
    thinking: Optional[str] = None
    tool_calls: list[dict] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseProvider(ABC):
    """
    Abstract base for AI model providers.

    All providers return the same ProviderResponse format.
    Agent code works identically regardless of provider.
    """

    def __init__(self, api_key: str = "", model: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ThinkingConfig] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Send messages to the AI model and get a response.

        Args:
            messages: Conversation history.
            tools: Available tools in provider format.
            thinking: Thinking configuration (if supported).

        Returns:
            Standardized ProviderResponse.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'claude', 'openai')."""
        ...

    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider/model supports extended thinking."""
        ...
