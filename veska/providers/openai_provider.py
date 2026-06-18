"""
OpenAI API provider.

Handles all communication with OpenAI's GPT models.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import openai

from veska.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    ThinkingConfig,
)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's GPT API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 8096,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(api_key=key, model=model, **kwargs)
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(api_key=key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def supports_thinking(self) -> bool:
        # OpenAI doesn't support extended thinking in the same way
        return False

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ThinkingConfig] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send messages to OpenAI and get a standardized response."""

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                openai_messages.append(assistant_msg)
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build API call kwargs
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }

        if tools:
            api_kwargs["tools"] = tools

        # Make the API call
        response = await self.client.chat.completions.create(**api_kwargs)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ProviderResponse:
        """Parse OpenAI's response into standardized format."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return ProviderResponse(
            content=content,
            thinking=None,  # OpenAI doesn't support thinking
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            stop_reason=choice.finish_reason,
        )
