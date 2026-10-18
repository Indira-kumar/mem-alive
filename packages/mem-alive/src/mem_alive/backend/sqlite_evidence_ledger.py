"""Append-only SQLite ledger for source events and their evidence."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ..schema.evidence_schema import Evidence, SourceEvent


class SQLiteEvidenceLedger:
    """Persist immutable sources and evidence with replay-safe IDs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS source_events (
                    id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    session_id TEXT,
                    PRIMARY KEY (namespace, id)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    span_start INTEGER,
                    span_end INTEGER,
                    PRIMARY KEY (namespace, id),
                    FOREIGN KEY (namespace, source_event_id)
                        REFERENCES source_events(namespace, id)
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS evidence_source_id "
                "ON evidence(namespace, source_event_id)"
            )

    def append_source(self, event: SourceEvent) -> bool:
        """Insert once; return False for an identical replay, reject changed payloads."""
        if not event.id or not event.namespace or not event.source_type:
            raise ValueError("source ID, namespace, and type are required")
        if event.occurred_at.utcoffset() is None:
            raise ValueError("source time must have a timezone")
        with closing(self._connect()) as connection, connection:
            inserted = connection.execute(
                """INSERT INTO source_events
                (id, namespace, source_type, content, occurred_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, id) DO NOTHING""",
                (
                    event.id,
                    event.namespace,
                    event.source_type,
                    event.content,
                    event.occurred_at.isoformat(),
                    event.session_id,
                ),
            ).rowcount
            if not inserted and self._read_source(connection, event.namespace, event.id) != event:
                raise ValueError(f"source ID {event.id!r} already has different content")
            return bool(inserted)

    def add_evidence(self, item: Evidence) -> bool:
        """Link evidence to an existing source; optionally locate its exact span."""
        if not item.id or not item.namespace or not item.kind or not item.content:
            raise ValueError("evidence ID, namespace, kind, and content are required")
        with closing(self._connect()) as connection, connection:
            source = self._read_source(connection, item.namespace, item.source_event_id)
            if source is None:
                raise ValueError(f"unknown source event {item.source_event_id!r}")
            start, end = item.span_start, item.span_end
            if (start is None) != (end is None):
                raise ValueError("evidence span must be within its source")
            if start is not None and end is not None and (
                type(start) is not int or type(end) is not int
            ):
                raise ValueError("evidence span offsets must be integers")
            if start is not None and end is not None and not (
                0 <= start < end <= len(source.content)
            ):
                raise ValueError("evidence span must be within its source")
            inserted = connection.execute(
                """INSERT INTO evidence
                (id, namespace, source_event_id, kind, content, span_start, span_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, id) DO NOTHING""",
                (
                    item.id,
                    item.namespace,
                    item.source_event_id,
                    item.kind,
                    item.content,
                    item.span_start,
                    item.span_end,
                ),
            ).rowcount
            if not inserted and self._read_evidence(connection, item.namespace, item.id) != item:
                raise ValueError(f"evidence ID {item.id!r} already has different content")
            return bool(inserted)

    def get_source(self, namespace: str, event_id: str) -> SourceEvent | None:
        """Read the original source event by ID."""
        with closing(self._connect()) as connection:
            return self._read_source(connection, namespace, event_id)

    def get_evidence(self, namespace: str, evidence_id: str) -> Evidence | None:
        """Read evidence and its source link by ID."""
        with closing(self._connect()) as connection:
            return self._read_evidence(connection, namespace, evidence_id)

    def evidence_for_source(self, namespace: str, event_id: str) -> list[Evidence]:
        """List evidence derived from one event in stable ID order."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE namespace = ? AND source_event_id = ? ORDER BY id",
                (namespace, event_id),
            ).fetchall()
            return [self._evidence_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _read_source(
        connection: sqlite3.Connection, namespace: str, event_id: str
    ) -> SourceEvent | None:
        row = connection.execute(
            "SELECT * FROM source_events WHERE namespace = ? AND id = ?", (namespace, event_id)
        ).fetchone()
        if row is None:
            return None
        return SourceEvent(
            id=row["id"],
            namespace=row["namespace"],
            source_type=row["source_type"],
            content=row["content"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            session_id=row["session_id"],
        )

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            namespace=row["namespace"],
            source_event_id=row["source_event_id"],
            kind=row["kind"],
            content=row["content"],
            span_start=row["span_start"],
            span_end=row["span_end"],
        )

    def _read_evidence(
        self, connection: sqlite3.Connection, namespace: str, evidence_id: str
    ) -> Evidence | None:
        row = connection.execute(
            "SELECT * FROM evidence WHERE namespace = ? AND id = ?", (namespace, evidence_id)
        ).fetchone()
        return self._evidence_from_row(row) if row is not None else None
