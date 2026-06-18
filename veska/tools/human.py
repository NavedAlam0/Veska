"""
Human-in-the-Loop tool.

Allows agents to pause mid-task and ask the human a question.
Developer provides the callback — we just call it.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Coroutine, Optional, Union

from veska.tools.base import Tool, ToolParameter

DEFAULT_TIMEOUT = 300  # 5 minutes


def create_ask_user_tool(
    callback: Callable[..., Union[str, Coroutine[Any, Any, str]]],
    timeout: int = DEFAULT_TIMEOUT,
) -> Tool:
    """
    Create an ask_user tool that calls the developer's callback.

    Args:
        callback: Function that takes a question string and returns the user's answer.
                  Can be sync or async.
        timeout: Max seconds to wait for user response.
    """

    async def ask_user(question: str) -> str:
        try:
            if inspect.iscoroutinefunction(callback):
                answer = await asyncio.wait_for(callback(question), timeout=timeout)
            else:
                answer = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, callback, question),
                    timeout=timeout,
                )
            return answer
        except asyncio.TimeoutError:
            return "Error: User did not respond within the time limit."
        except Exception as e:
            return f"Error: Failed to get user input: {e}"

    return Tool(
        name="ask_user",
        description="Ask the user a question when you need clarification or approval.",
        when_to_use=(
            "When you are unsure about a decision, need user preference, "
            "or want confirmation before a risky action."
        ),
        parameters=[
            ToolParameter(
                name="question",
                type="string",
                description="The question to ask the user",
                required=True,
            ),
        ],
        function=ask_user,
    )
