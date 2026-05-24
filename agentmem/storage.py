"""
storage.py — SQLite-backed persistence layer for AgentMem.
All memories live in a single local .db file. No cloud. No setup.
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Optional


class MemoryStorage:
    """
    Handles all SQLite read/write operations.
    Schema:
        memories(id, content, metadata, embedding_blob, created_at, updated_at, agent_id)
    """

    def __init__(self, db_path: str = "agentmem.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                agent_id    TEXT NOT NULL DEFAULT 'default',
                content     TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                embedding   BLOB,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent ON memories(agent_id)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(
        self,
        content: str,
        embedding: bytes,
        agent_id: str = "default",
        metadata: Optional[dict] = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO memories (id, agent_id, content, metadata, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                agent_id,
                content,
                json.dumps(metadata or {}),
                embedding,
                now,
                now,
            ),
        )
        self._conn.commit()
        return memory_id

    def update(self, memory_id: str, content: str, embedding: bytes, metadata: Optional[dict] = None):
        now = time.time()
        self._conn.execute(
            """
            UPDATE memories SET content=?, embedding=?, metadata=?, updated_at=?
            WHERE id=?
            """,
            (content, embedding, json.dumps(metadata or {}), now, memory_id),
        )
        self._conn.commit()

    def delete(self, memory_id: str):
        self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.commit()

    def clear(self, agent_id: str = "default"):
        self._conn.execute("DELETE FROM memories WHERE agent_id=?", (agent_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self, agent_id: str = "default") -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE agent_id=? ORDER BY created_at DESC",
            (agent_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_id(self, memory_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        return d

    def close(self):
        self._conn.close()
