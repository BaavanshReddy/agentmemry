# AgentMem 🧠

**Local-first memory for AI agents. No cloud. No setup. Just SQLite.**

[![PyPI version](https://badge.fury.io/py/agentmem.svg)](https://badge.fury.io/py/agentmem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Most agent memory frameworks require Postgres, Neo4j, or a cloud vector database. **AgentMem doesn't.** It runs entirely on your machine using SQLite and local embeddings — making it the fastest way to add persistent memory to any Python agent.

---

## Why AgentMem?

| Feature | AgentMem | Mem0 | Zep | Letta |
|---|---|---|---|---|
| Local-first (no cloud DB) | ✅ | ❌ | ❌ | ❌ |
| Zero setup | ✅ | ⚠️ | ❌ | ❌ |
| No API key required | ✅ | ❌ | ❌ | ✅ |
| Semantic search | ✅ | ✅ | ✅ | ✅ |
| MMR diversity search | ✅ | ❌ | ❌ | ❌ |
| Multi-agent isolation | ✅ | ✅ | ✅ | ✅ |
| Exportable / portable | ✅ (single .db file) | ❌ | ❌ | ❌ |

---

## Installation

```bash
pip install agentmem
```

That's it. No Docker. No database server. No API keys.

---

## Quickstart

```python
from agentmem import Memory

mem = Memory()

# Store memories
mem.add("User prefers concise, bullet-point answers")
mem.add("The project uses FastAPI and PostgreSQL")
mem.add("Authentication uses JWT tokens")

# Search
results = mem.search("what database are we using?")
print(results[0]["content"])
# → "The project uses FastAPI and PostgreSQL"

# Inject context directly into a system prompt
context = mem.get_context("how should I format my response?")
print(context)
# → Relevant memory: User prefers concise, bullet-point answers
```

---

## Core API

### `Memory(agent_id, db_path, model, top_k, threshold)`

| Parameter | Default | Description |
|---|---|---|
| `agent_id` | `"default"` | Namespace — different agents sharing one DB stay isolated |
| `db_path` | `"agentmem.db"` | Path to the SQLite file (created automatically) |
| `model` | `"all-MiniLM-L6-v2"` | sentence-transformers model for local embeddings |
| `top_k` | `5` | Default results returned by `search()` |
| `threshold` | `0.15` | Minimum similarity score (0–1) |

---

### `mem.add(content, metadata=None) → str`

Store a memory. Returns the memory ID.

```python
mem.add("Deadline is Friday", metadata={"type": "task", "priority": "high"})
```

---

### `mem.search(query, top_k=None, threshold=None, diverse=False) → list[dict]`

Semantic search over stored memories.

```python
results = mem.search("when is the deadline?")
for r in results:
    print(r["content"], r["score"])
```

Set `diverse=True` to use **Maximal Marginal Relevance** (MMR) — returns a more varied set of results instead of near-duplicates.

---

### `mem.get_context(query, top_k=None, prefix="Relevant memory: ") → str`

Returns a formatted string ready to inject into any LLM prompt.

```python
system_prompt = f"""
You are a helpful assistant.

{mem.get_context(user_message)}

Answer the user's question below.
"""
```

---

### Other methods

```python
mem.update(memory_id, new_content)   # Re-embed and overwrite
mem.delete(memory_id)                # Remove one memory
mem.clear()                          # Wipe all memories for this agent
mem.get_all()                        # List every memory (newest first)
mem.get_by_id(memory_id)             # Fetch a specific memory
mem.stats()                          # Count, db path, model info
```

---

## Use case: Coding agent with persistent project memory

```python
from agentmem import Memory

mem = Memory(agent_id="coding_assistant")

# Session 1 — onboarding
mem.add("This project uses Python 3.11 and FastAPI")
mem.add("Database is PostgreSQL with SQLAlchemy ORM")
mem.add("We use Alembic for migrations, not raw SQL")
mem.add("Tests are written with pytest, coverage must stay above 80%")

# Session 2 — new conversation, agent remembers everything
context = mem.get_context("How should I set up a new migration?")
# → Relevant memory: We use Alembic for migrations, not raw SQL
# → Relevant memory: Database is PostgreSQL with SQLAlchemy ORM
```

---

## Multi-agent support

Multiple agents can share the same database file without interfering:

```python
agent_a = Memory(agent_id="planner",   db_path="agents.db")
agent_b = Memory(agent_id="executor",  db_path="agents.db")
agent_c = Memory(agent_id="reviewer",  db_path="agents.db")
```

Each agent only sees its own memories.

---

## How it works

```
User input
    │
    ▼
sentence-transformers (local, offline)
    │  Converts text → float32 vector
    ▼
SQLite (agentmem.db)
    │  Stores content + embedding blob + metadata
    ▼
Cosine similarity search (numpy)
    │  Ranks stored memories by relevance
    ▼
Top-K results → injected into LLM prompt
```

No network calls. No external services. All computation happens in Python.

---

## Running the examples

```bash
# Clone the repo
git clone https://github.com/BaavanshReddy/agentmem.git
cd agentmem

# Install dependencies
pip install -e ".[dev]"

# Run quickstart
python examples/basic_usage.py

# Run coding agent example
python examples/coding_agent.py

# Run tests
pytest tests/ -v
```

---

## Requirements

- Python 3.9+
- sentence-transformers
- numpy

---

## Roadmap

- [ ] Automatic memory deduplication
- [ ] Time-decay scoring (older memories ranked lower)
- [ ] Memory compression for long-running agents
- [ ] LangChain / LlamaIndex integration
- [ ] Benchmark results on LongMemEval

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Baavansh Reddy Gundlapalli** — [GitHub](https://github.com/BaavanshReddy)
