"""
Claude API provider.

Handles all communication with Anthropic's Claude models.
Supports extended thinking and streaming.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional, Union

import anthropic

from veska.core.env import get_env
from veska.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    StreamEvent,
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

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 8096,
        **kwargs: Any,
    ) -> None:
        key = api_key or get_env("ANTHROPIC_API_KEY", "")
        resolved_model = model or get_env("ANTHROPIC_MODEL") or self.DEFAULT_MODEL

        super().__init__(api_key=key, model=resolved_model, **kwargs)
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
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncGenerator[StreamEvent, None]]:
        """Send messages to Claude. Returns full response or streams events."""
        api_kwargs = self._build_api_kwargs(messages, tools, thinking)

        if stream:
            return self._stream(api_kwargs, thinking)

        response = await self.client.messages.create(**api_kwargs)
        return self._parse_response(response, thinking)

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: Optional[list[dict]],
        thinking: Optional[ThinkingConfig],
    ) -> dict[str, Any]:
        """Convert messages and build API kwargs. Used by both bulk and stream."""
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
                content: list[dict[str, Any]] = []
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
                if isinstance(msg.content, list):
                    # Multi-modal content blocks
                    conversation.append({
                        "role": msg.role,
                        "content": _to_claude_content_blocks(msg.content),
                    })
                else:
                    conversation.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": conversation,
        }

        if system_prompt:
            api_kwargs["system"] = system_prompt

        if tools:
            api_kwargs["tools"] = tools

        if thinking and thinking.enabled and self.supports_thinking():
            api_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens,
            }

        return api_kwargs

    async def _stream(
        self,
        api_kwargs: dict[str, Any],
        thinking: Optional[ThinkingConfig],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response from Claude, yielding events as tokens arrive."""
        content_text = ""
        thinking_text = ""
        tool_calls = []
        current_tool: dict[str, Any] = {}
        current_tool_input = ""
        input_tokens = 0
        output_tokens = 0

        async with self.client.messages.stream(**api_kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        current_tool_input = ""
                        yield StreamEvent(
                            type="tool_call",
                            tool_name=block.name,
                            tool_call_id=block.id,
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        content_text += delta.text
                        yield StreamEvent(type="text_delta", text=delta.text)
                    elif delta.type == "thinking_delta":
                        thinking_text += delta.thinking
                        if thinking and thinking.output != "discard":
                            yield StreamEvent(type="thinking_delta", thinking=delta.thinking)
                    elif delta.type == "input_json_delta":
                        current_tool_input += delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            arguments = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        current_tool["arguments"] = arguments
                        tool_calls.append(current_tool)
                        current_tool = {}
                        current_tool_input = ""

                elif event.type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens

                elif event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

        yield StreamEvent(
            type="done",
            response=ProviderResponse(
                content=content_text,
                thinking=thinking_text if thinking_text and thinking and thinking.output != "discard" else None,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                stop_reason="end_turn" if not tool_calls else "tool_use",
            ),
        )

    def _parse_response(
        self,
        response: Any,
        thinking: Optional[ThinkingConfig],
    ) -> ProviderResponse:
        """Parse Claude's bulk response into standardized format."""
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


def _to_claude_content_blocks(blocks: list[dict]) -> list[dict]:
    """Convert processor content blocks to Claude API format."""
    claude_blocks = []

    for block in blocks:
        if block["type"] == "text":
            claude_blocks.append({"type": "text", "text": block["text"]})

        elif block["type"] == "image":
            if block.get("source_type") == "base64":
                claude_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block["media_type"],
                        "data": block["data"],
                    },
                })
            elif block.get("source_type") == "url":
                claude_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block["url"],
                    },
                })

        elif block["type"] == "document":
            if block.get("source_type") == "base64":
                claude_blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": block["media_type"],
                        "data": block["data"],
                    },
                })

    return claude_blocks
