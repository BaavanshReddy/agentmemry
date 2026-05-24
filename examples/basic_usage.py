"""
basic_usage.py — Quickstart example for AgentMem.

Run with:
    python examples/basic_usage.py
"""

from agentmem import Memory

# Create a memory store (creates agentmem.db in the current directory)
mem = Memory(agent_id="my_agent")

# Store some facts
mem.add("The user's name is Alex")
mem.add("Alex prefers concise, bullet-point answers")
mem.add("The project is a FastAPI backend with a React frontend")
mem.add("Alex dislikes long introductions in responses")
mem.add("Deployment target is AWS Lambda")

print(f"Stored 5 memories. Stats: {mem.stats()}\n")

# Search for relevant memories
query = "How should I format my next response?"
results = mem.search(query, top_k=3)

print(f"Query: '{query}'")
print(f"Top {len(results)} relevant memories:\n")
for r in results:
    print(f"  [{r['score']:.3f}] {r['content']}")

# Get a ready-to-use context string for LLM prompts
print("\n--- Context block (inject this into your system prompt) ---")
print(mem.get_context(query))

# Clean up (optional)
mem.clear()
print("\nMemories cleared.")
