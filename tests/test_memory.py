"""
test_memory.py — Core tests for AgentMem.

Run with:
    pytest tests/ -v
"""

import os
import pytest
from agentmem import Memory


TEST_DB = "test_agentmem.db"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test database before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def mem():
    return Memory(agent_id="test_agent", db_path=TEST_DB)


class TestAdd:
    def test_add_returns_id(self, mem):
        mid = mem.add("Python is a great language")
        assert isinstance(mid, str)
        assert len(mid) == 36  # UUID format

    def test_add_with_metadata(self, mem):
        mid = mem.add("Use FastAPI for REST", metadata={"source": "user"})
        stored = mem.get_by_id(mid)
        assert stored["metadata"]["source"] == "user"

    def test_multiple_adds(self, mem):
        for i in range(5):
            mem.add(f"Memory number {i}")
        assert mem.stats()["total_memories"] == 5


class TestSearch:
    def test_search_returns_results(self, mem):
        mem.add("The user prefers dark mode")
        mem.add("FastAPI is the backend framework")
        mem.add("Database is PostgreSQL")

        results = mem.search("what database are we using?")
        assert len(results) > 0
        assert results[0]["content"] == "Database is PostgreSQL"

    def test_search_has_score(self, mem):
        mem.add("Python 3.11 is required")
        results = mem.search("Python version")
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_search_top_k(self, mem):
        for i in range(10):
            mem.add(f"Fact number {i} about the project")
        results = mem.search("project fact", top_k=3)
        assert len(results) <= 3

    def test_search_empty_store(self, mem):
        results = mem.search("anything")
        assert results == []


class TestGetContext:
    def test_get_context_returns_string(self, mem):
        mem.add("User likes concise answers")
        context = mem.get_context("how should I respond?")
        assert isinstance(context, str)

    def test_get_context_empty_store(self, mem):
        context = mem.get_context("anything")
        assert context == ""

    def test_get_context_prefix(self, mem):
        mem.add("Use bullet points")
        context = mem.get_context("format", prefix="MEMORY: ")
        assert context.startswith("MEMORY: ")


class TestUpdateDelete:
    def test_update_content(self, mem):
        mid = mem.add("Old content")
        mem.update(mid, "New content")
        stored = mem.get_by_id(mid)
        assert stored["content"] == "New content"

    def test_delete(self, mem):
        mid = mem.add("Will be deleted")
        mem.delete(mid)
        assert mem.get_by_id(mid) is None

    def test_clear(self, mem):
        mem.add("Memory 1")
        mem.add("Memory 2")
        mem.clear()
        assert mem.stats()["total_memories"] == 0


class TestAgentIsolation:
    def test_agents_dont_share_memories(self):
        mem_a = Memory(agent_id="agent_a", db_path=TEST_DB)
        mem_b = Memory(agent_id="agent_b", db_path=TEST_DB)

        mem_a.add("Agent A secret")
        results = mem_b.search("Agent A secret")
        assert all("Agent A secret" not in r["content"] for r in results)

        mem_a.clear()
        mem_b.clear()


class TestStats:
    def test_stats_keys(self, mem):
        s = mem.stats()
        assert "agent_id" in s
        assert "total_memories" in s
        assert "db_path" in s
        assert "model" in s
