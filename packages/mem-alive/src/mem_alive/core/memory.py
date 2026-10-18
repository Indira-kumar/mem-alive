from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import List

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..ingestion.extractor import MemoryExtractor, Message
from ..schema.memory_schema import Memory as MemoryRecord
from ..schema.memory_schema import MemoryEvent, SearchResult
from ..store.episodic_store import EpisodicStore
from ..store.procedural_store import ProceduralStore
from ..store.semantic_store import SemanticStore


class Memory:
    def __init__(self, embedding_provider: EmbeddingProvider, db: StorageBackend):
        self._embedding_provider = embedding_provider
        self._db = db
        self._stores = {
            "procedural": ProceduralStore(db=db, embedding_provider=embedding_provider),
            "semantic": SemanticStore(db=db, embedding_provider=embedding_provider),
            "episodic": EpisodicStore(db=db, embedding_provider=embedding_provider),
        }

    async def recall(
        self,
        namespace: str,
        search_query: str,
        metadata: dict | None = None,
        top_k: int | None = None,
        memory_type: str | None = None,
    ) -> List[MemoryRecord]:
        """Return globally ranked memories across the requested stores."""
        hits = await self.recall_hits(namespace, search_query, metadata, top_k, memory_type)
        return [hit.memory for hit in hits]

    async def recall_hits(
        self,
        namespace: str,
        search_query: str,
        metadata: dict | None = None,
        top_k: int | None = None,
        memory_type: str | None = None,
    ) -> List[SearchResult]:
        """Return memories with their final store scores for inspection."""
        if not namespace.strip() or not search_query.strip():
            raise ValueError("namespace and search_query must be nonempty")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []
        metadata = metadata or {}
        if memory_type:
            return await self._stores[memory_type].recall_scored(
                namespace=namespace, search_query=search_query, metadata=metadata, top_k=top_k
            )
        results = await asyncio.gather(
            *(
                store.recall_scored(namespace, search_query, metadata, top_k)
                for store in self._stores.values()
            )
        )
        hits = [hit for group in results for hit in group]
        hits.sort(key=lambda hit: (-hit.score, hit.memory.id))
        return hits[:top_k]

    async def remember(
        self, memory_type: str, namespace: str, fact: str, metadata: dict | None = None
    ) -> MemoryRecord:
        """Store one already-extracted memory and return its ID-bearing record."""
        if memory_type not in self._stores:
            raise ValueError(f"unsupported memory type: {memory_type}")
        if not namespace.strip() or not fact.strip():
            raise ValueError("namespace and fact must be nonempty")
        return await self._stores[memory_type].remember(
            namespace=namespace, fact=fact, metadata=metadata or {}
        )

    async def get(self, namespace: str, memory_id: str) -> MemoryRecord | None:
        """Fetch a record inside its namespace, including superseded records."""
        return await self._db.get_memory_by_id(namespace, memory_id)

    async def update(
        self, namespace: str, memory_id: str, fact: str, metadata: dict | None = None
    ) -> MemoryRecord:
        """Explicitly correct a record by ID and refresh its embedding."""
        if not fact.strip():
            raise ValueError("fact must be nonempty")
        current = await self.get(namespace, memory_id)
        if current is None:
            raise KeyError(memory_id)
        vector = (await self._embedding_provider.embed([fact]))[0]
        updated = replace(
            current,
            content=fact,
            vector=vector,
            metadata=dict(current.metadata if metadata is None else metadata),
            updated_at=datetime.now(timezone.utc),
        )
        await self._db.upsert(updated)
        return updated

    async def delete(self, namespace: str, memory_id: str) -> None:
        """Remove an active record while preserving its audit history."""
        await self._db.delete(namespace, memory_id)

    async def forget(self, namespace: str, memory_id: str) -> None:
        """Purge a record and its local history from the given namespace."""
        await self._db.purge(namespace, memory_id)

    async def history(self, namespace: str, memory_id: str) -> List[MemoryEvent]:
        """Inspect add, update, and delete snapshots for one record."""
        return await self._db.history(namespace, memory_id)

    async def list(
        self,
        namespace: str,
        metadata: dict | None = None,
        memory_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """Browse active memories within one namespace, newest first."""
        return await self._db.list_memories(
            namespace, metadata or {}, memory_type, limit, offset
        )

    async def ingest(
        self,
        namespace: str,
        messages: Sequence[Message],
        extractor: MemoryExtractor,
        metadata: dict | None = None,
        store_source_text: bool = False,
    ) -> List[MemoryRecord]:
        """Extract memories, retaining source indices and hashes by default."""
        if not namespace.strip():
            raise ValueError("namespace must be nonempty")
        extracted = await extractor.extract(messages)
        for item in extracted:
            if (
                item.memory_type not in self._stores
                or not item.content.strip()
                or not item.source_indices
                or any(
                    index < 0
                    or index >= len(messages)
                    or messages[index]["role"] != "user"
                    for index in item.source_indices
                )
            ):
                raise ValueError("extractor returned an unsupported memory or source")
        saved = []
        seen = set()
        for item in extracted:
            key = (item.memory_type, item.content.strip().casefold())
            if key in seen:
                continue
            seen.add(key)
            existing = await self.recall(
                namespace, item.content, {}, top_k=10, memory_type=item.memory_type
            )
            if any(record.content.strip().casefold() == key[1] for record in existing):
                continue
            record_metadata = dict(metadata or {})
            record_metadata["_source"] = [
                {
                    "index": index,
                    "sha256": hashlib.sha256(messages[index]["content"].encode()).hexdigest(),
                    **({"text": messages[index]["content"]} if store_source_text else {}),
                }
                for index in item.source_indices
            ]
            saved.append(
                await self.remember(item.memory_type, namespace, item.content, record_metadata)
            )
        return saved
