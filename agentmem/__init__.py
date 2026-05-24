"""
AgentMem — Local-first memory for AI agents.
Zero cloud dependencies. SQLite + local embeddings.

Usage:
    from agentmem import Memory

    mem = Memory()
    mem.add("User prefers bullet-point answers")
    context = mem.get_context("How should I format this response?")
"""

from .memory import Memory

__version__ = "0.1.0"
__author__ = "Baavansh Reddy Gundlapalli"
__all__ = ["Memory"]
