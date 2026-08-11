"""Tests for explicit provider/model requirements."""

import pytest

from veska import Agent, Orchestrator


def test_agent_without_model_or_provider_raises_clear_error():
    with pytest.raises(ValueError, match="Agent requires either a model or provider"):
        Agent(name="assistant")


def test_agent_with_provider_none_raises_clear_error():
    with pytest.raises(ValueError, match="Agent requires either a model or provider"):
        Agent(name="assistant", provider=None)


def test_orchestrator_without_model_or_provider_raises_clear_error():
    with pytest.raises(ValueError, match="Orchestrator requires either a model or provider"):
        Orchestrator()


def test_orchestrator_with_provider_none_raises_clear_error():
    with pytest.raises(ValueError, match="Orchestrator requires either a model or provider"):
        Orchestrator(provider=None)
