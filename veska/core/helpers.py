"""
Helper utilities for Veska.

Centralizes provider resolution so Agent and Orchestrator
both use the same logic. Users never touch this file.
"""

from __future__ import annotations

from typing import Optional

from veska.providers.base import BaseProvider


# Model prefix to provider type mapping
CLAUDE_PREFIXES = ("claude",)
OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4")


def detect_provider_type(model: str) -> str:
    """Detect provider type from model name.

    Returns 'claude', 'openai', or raises ValueError.
    """
    model_lower = model.lower()

    for prefix in CLAUDE_PREFIXES:
        if model_lower.startswith(prefix):
            return "claude"

    for prefix in OPENAI_PREFIXES:
        if model_lower.startswith(prefix):
            return "openai"

    raise ValueError(
        f"Cannot detect provider for model '{model}'. "
        f"Model name must start with one of: "
        f"{', '.join(CLAUDE_PREFIXES + OPENAI_PREFIXES)}"
    )


def resolve_provider(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 8096,
    temperature: Optional[float] = None,
) -> BaseProvider:
    """Create the right provider from a model name and optional API key.

    - Detects provider type from model name
    - Reads API key from environment if not passed
    - Returns a ready-to-use provider instance
    """
    from veska.providers.claude_provider import ClaudeProvider
    from veska.providers.openai_provider import OpenAIProvider

    if model is None:
        model = "claude-sonnet-4-6"

    provider_type = detect_provider_type(model)

    if provider_type == "claude":
        return ClaudeProvider(api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature)
    elif provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature)
