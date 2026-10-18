"""Durable, local backend for small personal memory collections."""

import asyncio
import heapq
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..schema.memory_schema import Memory, MemoryEvent, SearchResult
from .storage_backend_interface import StorageBackend


class SQLiteBackend(StorageBackend):
    """Persist records in SQLite and score vectors exactly within a namespace.

    Search scans the namespace. This favors simple recovery and predictable scores
    over approximate indexing; larger collections will need another backend.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        with closing(self._connect()) as connection, connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError(f"unsupported SQLite memory schema version: {version}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT NOT NULL,
                    id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    superseded INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, id)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memory_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memory_events_lookup "
                "ON memory_events(namespace, id, sequence)"
            )
            connection.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            namespace=row["namespace"],
            content=row["content"],
            vector=json.loads(row["vector"]),
            metadata=json.loads(row["metadata"]),
            memory_type=row["memory_type"],
            superseded=bool(row["superseded"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _snapshot(memory: Memory) -> str:
        return json.dumps(
            {
                "id": memory.id,
                "namespace": memory.namespace,
                "content": memory.content,
                "vector": memory.vector,
                "metadata": memory.metadata,
                "memory_type": memory.memory_type,
                "superseded": memory.superseded,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            allow_nan=False,
        )

    @staticmethod
    def _record_event(connection: sqlite3.Connection, action: str, memory: Memory) -> None:
        connection.execute(
            "INSERT INTO memory_events(namespace, id, action, snapshot, occurred_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                memory.namespace,
                memory.id,
                action,
                SQLiteBackend._snapshot(memory),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("vector must be nonempty and finite")

    async def upsert(self, memory: Memory) -> None:
        self._validate_vector(memory.vector)
        vector_json = json.dumps(memory.vector, allow_nan=False)
        metadata_json = json.dumps(memory.metadata, allow_nan=False)

        def write() -> None:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT vector FROM memories WHERE namespace = ? AND id != ? LIMIT 1",
                    (memory.namespace, memory.id),
                ).fetchone()
                if row is not None and len(json.loads(row["vector"])) != len(memory.vector):
                    raise ValueError("memory vector dimension does not match namespace")
                existing = connection.execute(
                    "SELECT 1 FROM memories WHERE namespace = ? AND id = ?",
                    (memory.namespace, memory.id),
                ).fetchone()
                connection.execute(
                    """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, id) DO UPDATE SET
                    content=excluded.content, vector=excluded.vector,
                    metadata=excluded.metadata, memory_type=excluded.memory_type,
                    superseded=excluded.superseded, created_at=excluded.created_at,
                    updated_at=excluded.updated_at""",
                    (
                        memory.namespace,
                        memory.id,
                        memory.content,
                        vector_json,
                        metadata_json,
                        memory.memory_type,
                        int(memory.superseded),
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                    ),
                )
                self._record_event(connection, "update" if existing else "add", memory)

        await asyncio.to_thread(write)

    async def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        def read() -> Memory | None:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM memories WHERE namespace = ? AND id = ?", (namespace, id)
                ).fetchone()
                return self._from_row(row) if row is not None else None

        return await asyncio.to_thread(read)

    async def delete(self, namespace: str, id: str) -> None:
        def remove() -> None:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM memories WHERE namespace = ? AND id = ?", (namespace, id)
                ).fetchone()
                if row is None:
                    return
                connection.execute(
                    "DELETE FROM memories WHERE namespace = ? AND id = ?", (namespace, id)
                )
                self._record_event(connection, "delete", self._from_row(row))

        await asyncio.to_thread(remove)

    async def purge(self, namespace: str, id: str) -> None:
        def remove() -> None:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM memories WHERE namespace = ? AND id = ?", (namespace, id)
                )
                connection.execute(
                    "DELETE FROM memory_events WHERE namespace = ? AND id = ?", (namespace, id)
                )

        await asyncio.to_thread(remove)

    async def history(self, namespace: str, id: str) -> list[MemoryEvent]:
        def read() -> list[MemoryEvent]:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT action, snapshot, occurred_at FROM memory_events "
                    "WHERE namespace = ? AND id = ? ORDER BY sequence",
                    (namespace, id),
                ).fetchall()
            events = []
            for row in rows:
                snapshot = json.loads(row["snapshot"])
                snapshot["created_at"] = datetime.fromisoformat(snapshot["created_at"])
                snapshot["updated_at"] = datetime.fromisoformat(snapshot["updated_at"])
                events.append(
                    MemoryEvent(
                        action=row["action"],
                        memory=Memory(**snapshot),
                        occurred_at=datetime.fromisoformat(row["occurred_at"]),
                    )
                )
            return events

        return await asyncio.to_thread(read)

    async def list_memories(
        self,
        namespace: str,
        metadata: dict,
        memory_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")

        def read() -> list[Memory]:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT * FROM memories WHERE namespace = ? AND superseded = 0 "
                    "ORDER BY created_at DESC, id DESC",
                    (namespace,),
                ).fetchall()
            records = (self._from_row(row) for row in rows)
            filtered = [
                record
                for record in records
                if (memory_type is None or record.memory_type == memory_type)
                and all(record.metadata.get(key) == value for key, value in metadata.items())
            ]
            return filtered[offset : offset + limit]

        return await asyncio.to_thread(read)

    async def backup(self, destination: str | Path) -> Path:
        """Create a consistent SQLite snapshot at a new path."""
        target = Path(destination)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

        def copy() -> None:
            try:
                with closing(self._connect()) as source, closing(
                    sqlite3.connect(temporary)
                ) as output:
                    source.backup(output)
                temporary.chmod(0o600)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        await asyncio.to_thread(copy)
        return target

    async def search(
        self,
        namespace: str,
        vector: list[float],
        metadata: dict,
        top_k: int = 50,
        memory_type: str | None = None,
    ) -> list[SearchResult]:
        self._validate_vector(vector)
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        def retrieve() -> list[SearchResult]:
            with closing(self._connect()) as connection:
                if memory_type is None:
                    rows = connection.execute(
                        "SELECT * FROM memories WHERE namespace = ?", (namespace,)
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT * FROM memories WHERE namespace = ? AND memory_type = ?",
                        (namespace, memory_type),
                    ).fetchall()
            query_norm = math.sqrt(sum(value * value for value in vector))
            candidates = []
            for row in rows:
                memory = self._from_row(row)
                if not all(memory.metadata.get(key) == value for key, value in metadata.items()):
                    continue
                if len(memory.vector) != len(vector):
                    raise ValueError("query vector dimension does not match stored vectors")
                memory_norm = math.sqrt(sum(value * value for value in memory.vector))
                score = (
                    sum(left * right for left, right in zip(vector, memory.vector))
                    / (query_norm * memory_norm)
                    if query_norm and memory_norm
                    else 0.0
                )
                candidates.append(SearchResult(memory=memory, score=score))
            return heapq.nlargest(top_k, candidates, key=lambda result: result.score)

        return await asyncio.to_thread(retrieve)
