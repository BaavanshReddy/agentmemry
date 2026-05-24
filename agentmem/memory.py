"""
memory.py — The main AgentMem interface.
This is the only class most users will ever need to touch.

Quick start:
    from agentmem import Memory

    mem = Memory()
    mem.add("User prefers concise answers")
    mem.add("Project uses FastAPI and PostgreSQL")

    results = mem.search("what backend stack are we using?")
    context = mem.get_context("what backend stack are we using?")
"""

from __future__ import annotations

import time
from typing import Optional

from .storage import MemoryStorage
from .embeddings import embed, to_bytes, from_bytes
from .retrieval import search, mmr_search


class Memory:
    """
    Local-first agent memory.  All data lives in a SQLite file on disk.
    No API keys. No internet required. Works in any Python environment.

    Args:
        agent_id:   Namespace for this agent's memories. Use different IDs
                    for different agents sharing the same db file.
        db_path:    Path to the SQLite database file.
        model:      sentence-transformers model name for embeddings.
        top_k:      Default number of results returned by search().
        threshold:  Minimum similarity score (0–1) for search results.
    """

    def __init__(
        self,
        agent_id: str = "default",
        db_path: str = "agentmem.db",
        model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        threshold: float = 0.15,
    ):
        self.agent_id = agent_id
        self.model = model
        self.top_k = top_k
        self.threshold = threshold
        self._storage = MemoryStorage(db_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Store a new memory.

        Args:
            content:   The text to remember (a fact, preference, event, etc.)
            metadata:  Optional dict of extra fields (e.g. {"source": "user"}).

        Returns:
            The unique memory ID (UUID string).
        """
        vector = embed(content, self.model)
        blob = to_bytes(vector)
        memory_id = self._storage.insert(
            content=content,
            embedding=blob,
            agent_id=self.agent_id,
            metadata=metadata,
        )
        return memory_id

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        diverse: bool = False,
    ) -> list[dict]:
        """
        Retrieve the most relevant memories for a query.

        Args:
            query:      Natural-language query string.
            top_k:      Override the default number of results.
            threshold:  Override the default minimum similarity score.
            diverse:    If True, use Maximal Marginal Relevance to
                        return a more varied set of results.

        Returns:
            List of memory dicts (keys: id, content, metadata, score,
            created_at, updated_at).
        """
        k = top_k or self.top_k
        t = threshold if threshold is not None else self.threshold

        query_vec = embed(query, self.model)
        all_memories = self._storage.get_all(self.agent_id)

        if diverse:
            return mmr_search(query_vec, all_memories, top_k=k)
        return search(query_vec, all_memories, top_k=k, threshold=t)

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        prefix: str = "Relevant memory: ",
        separator: str = "\n",
    ) -> str:
        """
        Return a formatted string of relevant memories ready to inject
        into a system prompt or context window.

        Example output:
            Relevant memory: User prefers concise answers
            Relevant memory: Project uses FastAPI and PostgreSQL

        Args:
            query:     The current user message or task description.
            top_k:     Number of memories to include.
            prefix:    String prepended to each memory line.
            separator: String between memory lines.

        Returns:
            A plain-text block you can prepend to any LLM prompt.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        lines = [f"{prefix}{r['content']}" for r in results]
        return separator.join(lines)

    def update(self, memory_id: str, content: str, metadata: Optional[dict] = None):
        """
        Overwrite an existing memory's content and re-embed it.

        Args:
            memory_id: The ID returned by add().
            content:   New text content.
            metadata:  New metadata dict (replaces old one entirely).
        """
        vector = embed(content, self.model)
        blob = to_bytes(vector)
        self._storage.update(memory_id, content, blob, metadata)

    def delete(self, memory_id: str):
        """Remove a specific memory by ID."""
        self._storage.delete(memory_id)

    def clear(self):
        """Delete all memories for this agent."""
        self._storage.clear(self.agent_id)

    def get_all(self) -> list[dict]:
        """Return every memory for this agent, newest first."""
        return self._storage.get_all(self.agent_id)

    def get_by_id(self, memory_id: str) -> Optional[dict]:
        """Fetch a single memory by its ID."""
        return self._storage.get_by_id(memory_id)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return basic stats about the memory store."""
        all_mems = self.get_all()
        return {
            "agent_id": self.agent_id,
            "total_memories": len(all_mems),
            "db_path": str(self._storage.db_path),
            "model": self.model,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"Memory(agent_id={s['agent_id']!r}, "
            f"memories={s['total_memories']}, "
            f"db={s['db_path']!r})"
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._storage.close()
