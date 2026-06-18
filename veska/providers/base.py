"""
Base provider interface.

All AI model providers (Claude, OpenAI) implement this.
Agents don't know which provider they're using.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, list[dict]] = ""  # str for text, list for multi-modal
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


class StreamEvent:
    """A single event in a streaming response."""

    def __init__(
        self,
        type: str,
        text: str = "",
        tool_name: str = "",
        tool_arguments: Optional[dict] = None,
        tool_call_id: str = "",
        tool_result: str = "",
        thinking: str = "",
        response: Optional[ProviderResponse] = None,
        parsed_output: Optional[Any] = None,
    ) -> None:
        self.type = type  # "text_delta", "thinking_delta", "tool_call", "tool_result", "done"
        self.text = text
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {}
        self.tool_call_id = tool_call_id
        self.tool_result = tool_result
        self.thinking = thinking
        self.response = response
        self.parsed_output = parsed_output

    def __repr__(self) -> str:
        return f"StreamEvent(type={self.type})"


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
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ThinkingConfig] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Send messages to the AI model.

        Args:
            messages: Conversation history.
            tools: Available tools in provider format.
            thinking: Thinking configuration (if supported).
            stream: If True, returns AsyncGenerator[StreamEvent].
                    If False, returns ProviderResponse.
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
