"""
OpenAI API provider.

Handles all communication with OpenAI's GPT models.
Supports streaming.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional, Union

import openai

from veska.core.env import get_env
from veska.providers.base import (
    BaseProvider,
    Message,
    ProviderResponse,
    StreamEvent,
    ThinkingConfig,
)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's GPT API."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 8096,
        **kwargs: Any,
    ) -> None:
        key = api_key or get_env("OPENAI_API_KEY", "")
        resolved_model = model or get_env("OPENAI_MODEL") or self.DEFAULT_MODEL

        super().__init__(api_key=key, model=resolved_model, **kwargs)
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(api_key=key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def supports_thinking(self) -> bool:
        return False

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        thinking: Optional[ThinkingConfig] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncGenerator[StreamEvent, None]]:
        """Send messages to OpenAI. Returns full response or streams events."""
        api_kwargs = self._build_api_kwargs(messages, tools, stream)

        if stream:
            return self._stream(api_kwargs)

        response = await self.client.chat.completions.create(**api_kwargs)
        return self._parse_response(response)

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: Optional[list[dict]],
        stream: bool = False,
    ) -> dict[str, Any]:
        """Convert messages and build API kwargs. Used by both bulk and stream."""
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
                if isinstance(msg.content, list):
                    # Multi-modal content blocks
                    openai_messages.append({
                        "role": msg.role,
                        "content": _to_openai_content_blocks(msg.content),
                    })
                else:
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }

        if tools:
            api_kwargs["tools"] = tools

        if stream:
            api_kwargs["stream"] = True
            api_kwargs["stream_options"] = {"include_usage": True}

        return api_kwargs

    async def _stream(
        self,
        api_kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response from OpenAI, yielding events as tokens arrive."""
        content_text = ""
        tool_calls_data: dict[int, dict[str, Any]] = {}
        input_tokens = 0
        output_tokens = 0
        finish_reason = None

        stream = await self.client.chat.completions.create(**api_kwargs)

        async for chunk in stream:
            if not chunk.choices and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta and delta.content:
                content_text += delta.content
                yield StreamEvent(type="text_delta", text=delta.content)

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_data[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[idx]["name"] = tc_delta.function.name
                            yield StreamEvent(
                                type="tool_call",
                                tool_name=tc_delta.function.name,
                                tool_call_id=tc_delta.id or tool_calls_data[idx]["id"],
                            )
                        if tc_delta.function.arguments:
                            tool_calls_data[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = []
        for idx in sorted(tool_calls_data.keys()):
            tc = tool_calls_data[idx]
            try:
                arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": arguments,
            })

        yield StreamEvent(
            type="done",
            response=ProviderResponse(
                content=content_text,
                thinking=None,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                stop_reason=finish_reason,
            ),
        )

    def _parse_response(self, response: Any) -> ProviderResponse:
        """Parse OpenAI's bulk response into standardized format."""
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
            thinking=None,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            stop_reason=choice.finish_reason,
        )


def _to_openai_content_blocks(blocks: list[dict]) -> list[dict]:
    """Convert processor content blocks to OpenAI API format."""
    openai_blocks = []

    for block in blocks:
        if block["type"] == "text":
            openai_blocks.append({"type": "text", "text": block["text"]})

        elif block["type"] == "image":
            if block.get("source_type") == "base64":
                data_url = f"data:{block['media_type']};base64,{block['data']}"
                entry: dict = {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
                if block.get("detail") and block["detail"] != "auto":
                    entry["image_url"]["detail"] = block["detail"]
                openai_blocks.append(entry)

            elif block.get("source_type") == "url":
                entry = {
                    "type": "image_url",
                    "image_url": {"url": block["url"]},
                }
                if block.get("detail") and block["detail"] != "auto":
                    entry["image_url"]["detail"] = block["detail"]
                openai_blocks.append(entry)

        elif block["type"] == "document":
            # OpenAI doesn't support document blocks natively — include as text note
            openai_blocks.append({
                "type": "text",
                "text": "[PDF document attached — text extraction required for OpenAI]",
            })

    return openai_blocks
