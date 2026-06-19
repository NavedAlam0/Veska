"""Tests for the memory system — AgentMemory and SharedMemory."""

from veska.core.memory import AgentMemory, SharedMemory


def test_agent_memory_add_and_get_recent():
    """Agent should be able to store and retrieve recent memories."""
    mem = AgentMemory("test_agent")
    mem.add("api_key", "sk-123", category="config")

    recent = mem.get_recent(1)
    assert len(recent) == 1
    assert recent[0].value == "sk-123"


def test_agent_memory_get_all():
    """get_all should return all stored memories."""
    mem = AgentMemory("test_agent")
    mem.add("db_host", "localhost", category="config")
    mem.add("db_port", "5432", category="config")
    mem.add("user_name", "alice", category="user")

    all_items = mem.get_all()
    assert len(all_items) == 3


def test_agent_memory_categories():
    """get_by_category should return only memories in that category."""
    mem = AgentMemory("test_agent")
    mem.add("key1", "val1", category="a")
    mem.add("key2", "val2", category="b")
    mem.add("key3", "val3", category="a")

    a_items = mem.get_by_category("a")
    assert len(a_items) == 2


def test_agent_memory_errors():
    """Error tracking should work."""
    mem = AgentMemory("test_agent")
    mem.add_error("Something went wrong")
    mem.add_error("Another error")

    errors = mem.get_errors()
    assert len(errors) == 2


def test_shared_memory():
    """SharedMemory should allow cross-agent access."""
    shared = SharedMemory()
    mem1 = AgentMemory("agent1")
    mem2 = AgentMemory("agent2")

    mem1.add("result", "42", category="output")
    shared.store(mem1)
    shared.store(mem2)

    retrieved = shared.get("agent1")
    assert retrieved is not None
