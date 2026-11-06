"""Durable, namespace-isolated storage for small local memory collections.

SQLite owns transactions and audit history; exact cosine search runs over the
selected namespace. This deliberately favors correctness over approximate search.
"""

import asyncio
import json
import math
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from ..schema.memory_schema import Memory, SearchResult
from .storage_backend_interface import StorageBackend

T = TypeVar("T")


class ConcurrentWriteError(RuntimeError):
    """A reconciled memory changed before its mutation plan could commit."""


class SQLiteBackend(StorageBackend):
    """A local database with atomic writes and a durable mutation history.

    One database uses one embedding dimension. Re-embedding with another model
    requires a separate database. Supply ``embedding_model`` to bind its identity
    too. Binding an older unlabelled database trusts the caller to identify the
    model that originally embedded its records. Omitting the identity skips this
    check. Call ``aclose`` when the backend is no longer used.
    """

    def __init__(self, path: str | Path, *, embedding_model: str | None = None):
        if embedding_model is not None and not embedding_model.strip():
            raise ValueError("Embedding model identity must be nonempty")
        self._lock = threading.Lock()
        self._closed = False
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError(f"Unsupported memory database schema version: {version}")
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS memories ("
                    "namespace TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, "
                    "PRIMARY KEY(namespace, id))"
                )
                self._db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS memory_content_hash ON memories "
                    "(namespace, json_extract(payload, '$.content_hash')) "
                    "WHERE json_extract(payload, '$.content_hash') != ''"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value INTEGER)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS history (sequence INTEGER PRIMARY KEY, "
                    "namespace TEXT NOT NULL, id TEXT NOT NULL, event TEXT NOT NULL, "
                    "before_payload TEXT, after_payload TEXT, created_at TEXT NOT NULL)"
                )
                self._db.execute(
                    "CREATE INDEX IF NOT EXISTS history_scope ON history(namespace, id)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS context (namespace TEXT PRIMARY KEY, "
                    "summary TEXT NOT NULL)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS messages (sequence INTEGER PRIMARY KEY, "
                    "namespace TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)"
                )
                self._db.execute(
                    "CREATE INDEX IF NOT EXISTS messages_scope ON messages(namespace,sequence)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS entities (namespace TEXT NOT NULL, "
                    "name TEXT NOT NULL, vector TEXT NOT NULL, memory_id TEXT NOT NULL, "
                    "PRIMARY KEY(namespace,name,memory_id))"
                )
                self._db.execute(
                    "CREATE INDEX IF NOT EXISTS entities_memory ON entities(namespace,memory_id)"
                )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS revisions (namespace TEXT PRIMARY KEY, "
                    "revision INTEGER NOT NULL)"
                )
                self._db.execute(
                    "INSERT OR IGNORE INTO revisions SELECT namespace,MAX(sequence) "
                    "FROM history GROUP BY namespace"
                )
                if embedding_model is not None:
                    existing = self._db.execute(
                        "SELECT value FROM settings WHERE key='embedding_model'"
                    ).fetchone()
                    if existing is not None and str(existing[0]) != embedding_model:
                        raise ValueError("Embedding model differs from the database model")
                    self._db.execute(
                        "INSERT OR IGNORE INTO settings VALUES ('embedding_model', ?)",
                        (embedding_model,),
                    )
                self._db.execute("PRAGMA user_version=2")
        except BaseException:
            self._db.close()
            raise

    async def _run(self, operation: Callable[[], T]) -> T:
        def locked() -> T:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Memory database is closed")
                return operation()

        return await asyncio.to_thread(locked)

    async def aclose(self) -> None:
        """Close the connection after outstanding operations finish."""

        def close() -> None:
            with self._lock:
                if not self._closed:
                    self._db.close()
                    self._closed = True

        await asyncio.to_thread(close)

    async def __aenter__(self) -> "SQLiteBackend":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    @staticmethod
    def _encode(memory: Memory) -> str:
        if not memory.id or not memory.namespace or not memory.content.strip():
            raise ValueError("Memory id, namespace, and content must be nonempty")
        if memory.memory_type not in {"semantic", "episodic", "procedural"}:
            raise ValueError("Unknown memory type")
        SQLiteBackend._unit(memory.vector)
        payload = asdict(memory)
        for key in ("created_at", "updated_at"):
            timestamp = payload[key]
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("Memory timestamps must be timezone-aware")
            payload[key] = timestamp.isoformat()
        if not isinstance(memory.metadata, dict):
            raise ValueError("Memory metadata must be an object")
        return json.dumps(payload, allow_nan=False, sort_keys=True)

    @staticmethod
    def _decode(payload: str) -> Memory:
        values = json.loads(payload)
        for key in ("created_at", "updated_at"):
            values[key] = datetime.fromisoformat(values[key])
        return Memory(**values)

    @staticmethod
    def _unit(vector: list[float]) -> list[float]:
        if not vector or any(not math.isfinite(x) for x in vector):
            raise ValueError("Embedding must be nonempty and finite")
        norm = math.hypot(*vector)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("Embedding must have a finite nonzero norm")
        return [x / norm for x in vector]

    def _check_dimension(self, dimension: int, *, initialize: bool = False) -> None:
        row = self._db.execute("SELECT value FROM settings WHERE key='dimension'").fetchone()
        if row is not None and row[0] != dimension:
            raise ValueError(f"Embedding dimension {dimension} differs from database {row[0]}")
        if row is None and initialize:
            self._db.execute("INSERT INTO settings VALUES ('dimension', ?)", (dimension,))

    def _revision(self, namespace: str) -> int:
        row = self._db.execute(
            "SELECT revision FROM revisions WHERE namespace=?", (namespace,)
        ).fetchone()
        return row[0] if row else 0

    async def revision(self, namespace: str) -> int:
        """Return a mutation token to validate a later namespace reconciliation."""
        return await self._run(lambda: self._revision(namespace))

    async def apply_changes(
        self,
        upserts: list[Memory],
        deletes: list[tuple[str, str]],
        expected: list[Memory] | None = None,
        namespace_revision: tuple[str, int] | None = None,
        messages: list[dict] | None = None,
        context_namespace: str | None = None,
        summary: str | None = None,
        entities: dict[str, list[tuple[str, list[float]]]] | None = None,
    ) -> None:
        """Commit a complete mutation plan and its audit records atomically.

        Inputs are snapshotted before any database work. A failed validation or
        write rolls back the entire batch, including its history and dimension.
        Supply a revision captured before reading candidates to reject phantom
        inserts as well as stale updates from another backend instance.
        """
        encoded = [(m.namespace, m.id, len(m.vector), self._encode(m)) for m in upserts]
        deletions = list(deletes)
        snapshots = [(m.namespace, m.id, self._encode(m)) for m in (expected or [])]
        keys = [(ns, id) for ns, id, _, _ in encoded] + deletions
        if len(set(keys)) != len(keys):
            raise ValueError("A mutation batch may touch each memory only once")

        if (messages is not None or summary is not None) and not context_namespace:
            raise ValueError("Conversation context requires a namespace")
        message_rows = []
        for message in messages or []:
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant", "system", "tool"}:
                raise ValueError("Unknown message role")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content must be nonempty text")
            message_rows.append((context_namespace, role, content))
        if summary is not None and not isinstance(summary, str):
            raise ValueError("Summary must be text")
        entity_rows: dict[str, list[tuple[str, int, str]]] = {}
        for id, links in (entities or {}).items():
            if id not in {key[1] for key in encoded}:
                raise ValueError("Entity links must belong to an upserted memory")
            entity_rows[id] = []
            seen = set()
            for name, vector in links:
                if not isinstance(name, str) or not name.strip() or name in seen:
                    raise ValueError("Entity names must be nonempty and unique per memory")
                seen.add(name)
                self._unit(vector)
                entity_rows[id].append((name, len(vector), json.dumps(vector, allow_nan=False)))

        def apply() -> None:
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                changed: set[str] = set()
                if namespace_revision is not None:
                    scope, revision = namespace_revision
                    if self._revision(scope) != revision:
                        raise ConcurrentWriteError("Namespace changed during reconciliation; retry")
                for namespace, id, snapshot in snapshots:
                    current = self._db.execute(
                        "SELECT payload FROM memories WHERE namespace=? AND id=?", (namespace, id)
                    ).fetchone()
                    if current is None or self._encode(self._decode(current[0])) != snapshot:
                        raise ConcurrentWriteError("Memory changed during reconciliation; retry")
                for namespace, id, dimension, payload in encoded:
                    self._check_dimension(dimension, initialize=True)
                    old = self._db.execute(
                        "SELECT payload FROM memories WHERE namespace=? AND id=?", (namespace, id)
                    ).fetchone()
                    self._db.execute(
                        "INSERT INTO memories VALUES (?, ?, ?) "
                        "ON CONFLICT(namespace, id) DO UPDATE SET payload=excluded.payload",
                        (namespace, id, payload),
                    )
                    self._audit(namespace, id, "UPDATE" if old else "ADD", old, payload)
                    changed.add(namespace)
                    if id in entity_rows:
                        self._db.execute(
                            "DELETE FROM entities WHERE namespace=? AND memory_id=?",
                            (namespace, id),
                        )
                        for name, entity_dimension, vector_payload in entity_rows[id]:
                            self._check_dimension(entity_dimension, initialize=True)
                            self._db.execute(
                                "INSERT INTO entities VALUES (?, ?, ?, ?)",
                                (namespace, name, vector_payload, id),
                            )
                for namespace, id in deletions:
                    old = self._db.execute(
                        "SELECT payload FROM memories WHERE namespace=? AND id=?", (namespace, id)
                    ).fetchone()
                    if old:
                        self._db.execute(
                            "DELETE FROM memories WHERE namespace=? AND id=?", (namespace, id)
                        )
                        self._db.execute(
                            "DELETE FROM entities WHERE namespace=? AND memory_id=?",
                            (namespace, id),
                        )
                        self._audit(namespace, id, "DELETE", old, None)
                        changed.add(namespace)
                if message_rows:
                    self._db.executemany(
                        "INSERT INTO messages(namespace,role,content) VALUES (?, ?, ?)",
                        message_rows,
                    )
                    self._db.execute(
                        "DELETE FROM messages WHERE namespace=? AND sequence NOT IN "
                        "(SELECT sequence FROM messages WHERE namespace=? "
                        "ORDER BY sequence DESC LIMIT 10)",
                        (context_namespace, context_namespace),
                    )
                    changed.add(str(context_namespace))
                if summary is not None:
                    self._db.execute(
                        "INSERT INTO context VALUES (?, ?) ON CONFLICT(namespace) "
                        "DO UPDATE SET summary=excluded.summary",
                        (context_namespace, summary),
                    )
                    changed.add(str(context_namespace))
                for namespace in changed:
                    self._db.execute(
                        "INSERT INTO revisions VALUES (?, 1) ON CONFLICT(namespace) "
                        "DO UPDATE SET revision=revision+1",
                        (namespace,),
                    )

        await self._run(apply)

    def _audit(self, namespace: str, id: str, event: str, old, payload: str | None) -> None:
        self._db.execute(
            "INSERT INTO history(namespace,id,event,before_payload,after_payload,created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                namespace,
                id,
                event,
                old[0] if old else None,
                payload,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    async def upsert(self, memory: Memory) -> None:
        await self.apply_changes([memory], [])

    async def delete(self, namespace: str, id: str) -> None:
        await self.apply_changes([], [(namespace, id)])

    async def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        def get() -> Memory | None:
            row = self._db.execute(
                "SELECT payload FROM memories WHERE namespace=? AND id=?", (namespace, id)
            ).fetchone()
            return self._decode(row[0]) if row else None

        return await self._run(get)

    async def find_by_hash(self, namespace: str, digest: str) -> Memory | None:
        """Look up an exact content hash within a namespace."""
        if not digest:
            raise ValueError("Content digest must be nonempty")

        def get() -> Memory | None:
            row = self._db.execute(
                "SELECT payload FROM memories WHERE namespace=? AND "
                "json_extract(payload, '$.content_hash')=? AND "
                "json_extract(payload, '$.content_hash') != ''",
                (namespace, digest),
            ).fetchone()
            return self._decode(row[0]) if row else None

        return await self._run(get)

    async def recent_memories(self, namespace: str, limit: int = 10) -> list[Memory]:
        """Return recently updated records for extraction context."""
        if limit < 0:
            raise ValueError("limit must be nonnegative")

        def read() -> list[Memory]:
            rows = self._db.execute(
                "SELECT payload FROM memories WHERE namespace=? ORDER BY "
                "julianday(json_extract(payload, '$.updated_at')) DESC, "
                "json_extract(payload, '$.updated_at') DESC, id LIMIT ?",
                (namespace, limit),
            )
            return [self._decode(row[0]) for row in rows]

        return await self._run(read)

    def _memories(self, namespace: str, metadata: dict) -> list[Memory]:
        rows = self._db.execute(
            "SELECT payload FROM memories WHERE namespace=? ORDER BY id", (namespace,)
        )
        memories = [self._decode(row[0]) for row in rows]
        return [
            m
            for m in memories
            if all(k in m.metadata and m.metadata[k] == v for k, v in metadata.items())
        ]

    async def list_memories(
        self, namespace: str, metadata: dict | None = None, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """List a namespace deterministically by id, with exact metadata filters."""
        if limit < 0 or offset < 0:
            raise ValueError("Pagination values must be nonnegative")
        filters = dict(metadata or {})
        return await self._run(lambda: self._memories(namespace, filters)[offset : offset + limit])

    async def search(
        self, namespace: str, vector: list[float], metadata: dict, top_k: int = 50
    ) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        unit = self._unit(vector)
        filters = dict(metadata)

        def search() -> list[SearchResult]:
            self._check_dimension(len(unit))
            results = [
                SearchResult(m, sum(a * b for a, b in zip(unit, self._unit(m.vector))))
                for m in self._memories(namespace, filters)
            ]
            results.sort(key=lambda r: (-r.score, r.memory.id))
            return results[:top_k]

        return await self._run(search)

    async def recent_messages(self, namespace: str) -> list[dict[str, str]]:
        """Return up to ten conversation messages in chronological order."""

        def read() -> list[dict[str, str]]:
            rows = self._db.execute(
                "SELECT role,content FROM messages WHERE namespace=? ORDER BY sequence",
                (namespace,),
            )
            return [{"role": role, "content": content} for role, content in rows]

        return await self._run(read)

    async def get_summary(self, namespace: str) -> str:
        """Return the durable rolling summary, or an empty initial summary."""

        def read() -> str:
            row = self._db.execute(
                "SELECT summary FROM context WHERE namespace=?", (namespace,)
            ).fetchone()
            return row[0] if row else ""

        return await self._run(read)

    async def entity_search(
        self, namespace: str, vector: list[float], top_k: int = 20
    ) -> list[tuple[str, float, list[str]]]:
        """Rank distinct entity names by their best cosine match and return links."""
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        unit = self._unit(vector)

        def search() -> list[tuple[str, float, list[str]]]:
            self._check_dimension(len(unit))
            grouped: dict[str, tuple[float, list[str]]] = {}
            rows = self._db.execute(
                "SELECT name,vector,memory_id FROM entities WHERE namespace=? "
                "ORDER BY name,memory_id",
                (namespace,),
            )
            for name, payload, memory_id in rows:
                score = sum(a * b for a, b in zip(unit, self._unit(json.loads(payload))))
                previous, ids = grouped.get(name, (-math.inf, []))
                grouped[name] = (max(previous, score), ids + [memory_id])
            results = [(name, score, ids) for name, (score, ids) in grouped.items()]
            results.sort(key=lambda entry: (-entry[1], entry[0]))
            return results[:top_k]

        return await self._run(search)

    async def history(self, namespace: str, id: str) -> list[dict[str, Any]]:
        """Return the scoped audit trail, including content retained after deletion."""

        def history() -> list[dict[str, Any]]:
            rows = self._db.execute(
                "SELECT event,before_payload,after_payload,created_at FROM history "
                "WHERE namespace=? AND id=? ORDER BY sequence",
                (namespace, id),
            )
            return [
                {
                    "event": event,
                    "id": id,
                    "namespace": namespace,
                    "before": self._history_payload(before),
                    "after": self._history_payload(after),
                    "created_at": created,
                }
                for event, before, after, created in rows
            ]

        return await self._run(history)

    @staticmethod
    def _history_payload(payload: str | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        record = json.loads(payload)
        record.pop("vector", None)
        return record
