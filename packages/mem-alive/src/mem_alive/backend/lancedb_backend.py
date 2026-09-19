import asyncio
import json
from pathlib import Path
from typing import Any

from ..schema.memory_schema import Memory, SearchResult
from .storage_backend_interface import StorageBackend


class LanceDBBackend(StorageBackend):
    """Persistent LanceDB implementation of the storage contract.

    LanceDB is imported lazily so the base package remains usable without the
    optional ``lancedb`` extra.
    """

    def __init__(self, uri: str | Path, table_name: str = "memories"):
        self._uri = str(uri)
        self._table_name = table_name
        self._connection: Any | None = None
        self._table: Any | None = None
        self._connection_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def search(
        self, namespace: str, vector: list[float], metadata: dict, top_k: int = 50
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        table = await self._get_table()
        if table is None:
            return []

        namespace_filter = f"namespace = {self._sql_literal(namespace)}"
        candidate_count = await table.count_rows(namespace_filter)
        if candidate_count == 0:
            return []

        query = await table.search(vector, vector_column_name="vector")
        rows = (
            await query.distance_type("cosine")
            .where(namespace_filter)
            .limit(candidate_count)
            .to_arrow()
        ).to_pylist()
        results = [
            SearchResult(memory=self._to_memory(row), score=1.0 - float(row["_distance"]))
            for row in rows
            if self._matches_metadata(row, metadata)
        ]
        return results[:top_k]

    async def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        table = await self._get_table()
        if table is None:
            return None

        predicate = self._identity_filter(namespace, id)
        rows = (await table.query().where(predicate).limit(1).to_arrow()).to_pylist()
        return self._to_memory(rows[0]) if rows else None

    async def upsert(self, memory: Memory) -> None:
        row = self._to_row(memory)
        async with self._write_lock:
            table = await self._get_table()
            if table is None:
                table = await self._create_table(len(memory.vector))
            await self._validate_vector_size(table, len(memory.vector))
            operation = (
                table.merge_insert(["namespace", "id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
            )
            await operation.execute([row])

    async def delete(self, namespace: str, id: str) -> None:
        async with self._write_lock:
            table = await self._get_table()
            if table is not None:
                await table.delete(self._identity_filter(namespace, id))

    async def aclose(self) -> None:
        if self._table is not None:
            self._table.close()
            self._table = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()

    async def _get_connection(self):
        if self._connection is not None:
            return self._connection

        async with self._connection_lock:
            if self._connection is None:
                try:
                    import lancedb
                except ImportError as error:
                    raise ImportError(
                        "LanceDBBackend requires the optional dependency: "
                        "pip install 'mem-alive[lancedb]'"
                    ) from error
                self._connection = await lancedb.connect_async(self._uri)
        return self._connection

    async def _get_table(self):
        if self._table is not None:
            return self._table

        connection = await self._get_connection()
        table_names = (await connection.list_tables()).tables
        if self._table_name in table_names:
            self._table = await connection.open_table(self._table_name)
        return self._table

    async def _create_table(self, vector_size: int):
        import pyarrow as pa

        schema = pa.schema(
            [
                ("id", pa.string()),
                ("namespace", pa.string()),
                ("content", pa.string()),
                ("vector", pa.list_(pa.float32(), vector_size)),
                ("metadata", pa.string()),
                ("memory_type", pa.string()),
                ("superseded", pa.bool_()),
                ("created_at", pa.timestamp("us", tz="UTC")),
                ("updated_at", pa.timestamp("us", tz="UTC")),
            ]
        )
        connection = await self._get_connection()
        self._table = await connection.create_table(self._table_name, schema=schema, exist_ok=True)
        return self._table

    @staticmethod
    async def _validate_vector_size(table, vector_size: int) -> None:
        schema = await table.schema()
        expected_size = schema.field("vector").type.list_size
        if vector_size != expected_size:
            raise ValueError(
                f"Embedding dimension mismatch: table expects {expected_size}, got {vector_size}"
            )

    @staticmethod
    def _to_row(memory: Memory) -> dict:
        return {
            "id": memory.id,
            "namespace": memory.namespace,
            "content": memory.content,
            "vector": memory.vector,
            "metadata": json.dumps(memory.metadata, sort_keys=True, separators=(",", ":")),
            "memory_type": memory.memory_type,
            "superseded": memory.superseded,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }

    @staticmethod
    def _to_memory(row: dict) -> Memory:
        return Memory(
            id=row["id"],
            namespace=row["namespace"],
            content=row["content"],
            vector=[float(value) for value in row["vector"]],
            metadata=json.loads(row["metadata"]),
            memory_type=row["memory_type"],
            superseded=row["superseded"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _matches_metadata(row: dict, metadata: dict) -> bool:
        stored_metadata = json.loads(row["metadata"])
        return all(stored_metadata.get(key) == value for key, value in metadata.items())

    @staticmethod
    def _identity_filter(namespace: str, id: str) -> str:
        return (
            f"namespace = {LanceDBBackend._sql_literal(namespace)} "
            f"AND id = {LanceDBBackend._sql_literal(id)}"
        )

    @staticmethod
    def _sql_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
