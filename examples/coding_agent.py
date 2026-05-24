"""
coding_agent.py — AgentMem used in a coding agent context.

Simulates a coding assistant that remembers project decisions, preferences,
and context across multiple sessions — without any cloud setup.

Run with:
    python examples/coding_agent.py
"""

from agentmem import Memory

# Separate agent_id for the coding assistant
mem = Memory(agent_id="coding_assistant", db_path="coding_agent.db")

# Simulate session 1: user tells the agent about the project
print("=== Session 1: Setting up project context ===\n")

session_1_facts = [
    "This project uses Python 3.11 and FastAPI",
    "Database is PostgreSQL with SQLAlchemy ORM",
    "We decided to use Alembic for migrations, not raw SQL",
    "The API follows REST conventions, no GraphQL",
    "Authentication uses JWT tokens, not sessions",
    "Tests are written with pytest, coverage must stay above 80%",
    "We avoid using global state; all dependencies injected via FastAPI Depends()",
    "The main entry point is app/main.py",
]

for fact in session_1_facts:
    mem.add(fact, metadata={"session": 1, "type": "project_decision"})

print(f"Stored {len(session_1_facts)} project decisions.\n")

# Simulate session 2: new conversation, agent needs context
print("=== Session 2: New conversation, agent retrieves context ===\n")

questions = [
    "How should I handle database migrations?",
    "What testing framework should I use?",
    "Where is the main entry point of the app?",
    "Should I use sessions or tokens for auth?",
]

for q in questions:
    context = mem.get_context(q, top_k=2)
    print(f"Q: {q}")
    print(f"Context injected:\n{context}\n")

# Show stats
print("=== Memory Stats ===")
print(mem.stats())

# Clean up
mem.clear()
