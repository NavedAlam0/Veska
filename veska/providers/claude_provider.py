"""
Claude API provider.

Handles all communication with Anthropic's Claude models.
Supports extended thinking for models that support it.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import anthropic

from veska.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    ThinkingConfig,
)

# Models that support extended thinking
THINKING_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250918",
}


class ClaudeProvider(BaseProvider):
    """Provider for Anthropic's Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8096,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        super().__init__(api_key=key, model=model, **kwargs)
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic(api_key=key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def supports_thinking(self) -> bool:
        return self.model in THINKING_MODELS

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ThinkingConfig] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send messages to Claude and get a standardized response."""

        # Separate system message from conversation
        system_prompt = ""
        conversation = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "tool":
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                conversation.append({"role": "assistant", "content": content})
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build API call kwargs
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": conversation,
        }

        if system_prompt:
            api_kwargs["system"] = system_prompt

        if tools:
            api_kwargs["tools"] = tools

        # Handle thinking
        use_thinking = (
            thinking
            and thinking.enabled
            and self.supports_thinking()
        )

        if use_thinking:
            api_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens,
            }

        # Make the API call
        response = await self.client.messages.create(**api_kwargs)

        # Parse response
        return self._parse_response(response, thinking)

    def _parse_response(
        self,
        response: Any,
        thinking: Optional[ThinkingConfig],
    ) -> ProviderResponse:
        """Parse Claude's response into standardized format."""
        content = ""
        thinking_text = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "thinking":
                if thinking and thinking.output != "discard":
                    thinking_text = block.thinking
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return ProviderResponse(
            content=content,
            thinking=thinking_text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )
