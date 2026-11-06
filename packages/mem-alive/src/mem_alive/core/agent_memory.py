"""Conversation memory: extract, retrieve, reconcile, then commit atomically."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from uuid import uuid4

from ..backend.sqlite_backend import SQLiteBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..inference import InferenceProvider
from ..retrieval import lemmas, rerank
from ..schema.memory_schema import Memory as Record


def serialize(record: Record) -> dict:
    """Public memory representation deliberately excludes the embedding."""
    result = asdict(record)
    result.pop("vector")
    result["created_at"] = record.created_at.isoformat()
    result["updated_at"] = record.updated_at.isoformat()
    return result


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _limit(value: int, maximum: int = 1000) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")


class AgentMemory:
    """Async application API shared by Python, MCP, and HTTP adapters.

    SQLite owns atomic persistence and history. Inference is an injected policy.
    Recent context and entity links persist
    across sessions; retrieval only reranks semantic candidates.
    Separate instances may race: stale updates fail rather than overwrite data.
    """

    def __init__(
        self,
        db: SQLiteBackend,
        embedding_provider: EmbeddingProvider,
        inference: InferenceProvider,
    ) -> None:
        self.db = db
        self.embeddings = embedding_provider
        self.inference = inference
        self._write_lock = asyncio.Lock()

    async def _embed(self, texts: builtins.list[str]) -> builtins.list[builtins.list[float]]:
        vectors = await self.embeddings.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding provider returned the wrong number of vectors")
        dimension = len(vectors[0]) if vectors else 0
        for vector in vectors:
            if (
                not dimension
                or len(vector) != dimension
                or not all(math.isfinite(v) for v in vector)
                or not any(vector)
            ):
                raise ValueError("embedding provider returned invalid vectors")
        return vectors

    async def _entity_links(self, records: builtins.list[Record]) -> dict:
        if not records:
            return {}
        groups = await self.inference.extract_entities([record.content for record in records])
        if len(groups) != len(records):
            raise ValueError("entity extraction response does not match memories")
        groups = [[_text(name, "entity") for name in group] for group in groups]
        names = list(dict.fromkeys(name for group in groups for name in group))
        vectors = await self._embed(names) if names else []
        by_name = dict(zip(names, vectors))
        return {
            record.id: [(name, by_name[name]) for name in dict.fromkeys(group)]
            for record, group in zip(records, groups)
        }

    async def add(
        self,
        messages: builtins.list[dict[str, str]],
        namespace: str,
        metadata: dict | None = None,
        infer: bool = True,
    ) -> builtins.list[dict]:
        """Extract with recent context, deduplicate exact text, atomically add facts.

        New facts accumulate, including corrections. Use update/delete for explicit
        changes. infer=False stores raw text without any language-model calls.
        """
        namespace = _text(namespace, "namespace")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
            raise ValueError("messages must contain between 1 and 100 turns")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"user", "assistant", "system", "tool"}
            ):
                raise ValueError("each message needs a supported role and content")
            _text(message["content"], "message content")
        if not isinstance(infer, bool):
            raise ValueError("infer must be a boolean")
        metadata = dict(metadata or {})
        # Validate JSON before model work; caller metadata never owns internal fields.
        json.dumps(metadata, allow_nan=False)
        async with self._write_lock:
            revision = await self.db.revision(namespace)
            summary = await self.db.get_summary(namespace)
            if infer:
                flattened = "\n".join(message["content"] for message in messages)
                # Bound only the retrieval window; extraction receives every input turn.
                query_vector = (await self._embed([flattened[-16_000:]]))[0]
                relevant = await self.db.search(namespace, query_vector, {}, top_k=10)
                recent = await self.db.recent_memories(namespace, limit=10)
                context = {
                    "summary": summary,
                    "recent_messages": await self.db.recent_messages(namespace),
                    "relevant_memories": [
                        r.memory.content for r in relevant if not r.memory.superseded
                    ],
                    "recent_memories": [r.content for r in recent],
                    "current_date": datetime.now(timezone.utc).isoformat(),
                    "conversation_date": metadata.get("conversation_date"),
                }
                facts = await self.inference.extract(messages, context=context)
                summary = await self.inference.summarize(summary, messages)
            else:
                facts = [message["content"] for message in messages]
            if len(facts) > 100:
                raise ValueError("extraction exceeded 100 facts")
            writes: builtins.list[Record] = []
            events: builtins.list[dict] = []
            # Exact wording, including whitespace, defines duplicate identity.
            for fact in dict.fromkeys(facts):
                _text(fact, "fact")
                digest = hashlib.sha256(fact.encode()).hexdigest()
                existing = await self.db.find_by_hash(namespace, digest)
                if existing is not None:
                    events.append({"event": "NONE", "memory": serialize(existing)})
                    continue
                writes.append(
                    Record(
                        id=str(uuid4()),
                        content=fact,
                        vector=[],
                        namespace=namespace,
                        metadata=metadata,
                        memory_type="semantic",
                        content_hash=digest,
                        keywords=lemmas(fact),
                    )
                )
            if writes:
                vectors = await self._embed([record.content for record in writes])
                for record, vector in zip(writes, vectors):
                    record.vector = vector
            entities = await self._entity_links(writes) if infer else {}
            await self.db.apply_changes(
                writes,
                [],
                namespace_revision=(namespace, revision),
                messages=messages,
                context_namespace=namespace,
                summary=summary,
                entities=entities,
            )
            return events + [{"event": "ADD", "memory": serialize(record)} for record in writes]

    async def search(
        self,
        query: str,
        namespace: str,
        limit: int = 10,
        metadata: dict | None = None,
        threshold: float = 0.3,
    ) -> builtins.list[dict]:
        """Rerank max(4*K,60) semantic candidates with BM25 and entity boosts."""
        namespace = _text(namespace, "namespace")
        query = _text(query, "query")
        _limit(limit)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        vector = (await self._embed([query]))[0]
        candidates = await self.db.search(
            namespace, vector, metadata or {}, top_k=max(4 * limit, 60)
        )
        candidates = [hit for hit in candidates if not hit.memory.superseded]
        if not candidates:
            return []
        groups = await self.inference.extract_entities([query])
        if len(groups) != 1:
            raise ValueError("query entity response must contain one group")
        names = list(dict.fromkeys(_text(name, "query entity") for name in groups[0]))[:8]
        entity_hits = []
        if names:
            for entity_vector in await self._embed(names):
                entity_hits.extend(await self.db.entity_search(namespace, entity_vector, top_k=500))
        ranked = rerank(query, candidates, entity_hits)
        return [
            {**serialize(hit.memory), **components}
            for hit, components in ranked
            if components["score"] >= threshold
        ][:limit]

    async def get(self, memory_id: str, namespace: str) -> dict | None:
        record = await self.db.get_memory_by_id(_text(namespace, "namespace"), memory_id)
        return serialize(record) if record else None

    async def list(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
        metadata: dict | None = None,
    ) -> builtins.list[dict]:
        _limit(limit)
        return [
            serialize(record)
            for record in await self.db.list_memories(
                _text(namespace, "namespace"), metadata=metadata, limit=limit, offset=offset
            )
        ]

    async def update(self, memory_id: str, content: str, namespace: str) -> dict:
        namespace = _text(namespace, "namespace")
        content = _text(content, "content")
        async with self._write_lock:
            record = await self.db.get_memory_by_id(namespace, memory_id)
            if record is None:
                raise KeyError(memory_id)
            vector = (await self._embed([content]))[0]
            updated = replace(
                record,
                content=content,
                vector=vector,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                keywords=lemmas(content),
                updated_at=datetime.now(timezone.utc),
            )
            entities = await self._entity_links([updated])
            await self.db.apply_changes([updated], [], expected=[record], entities=entities)
            return serialize(updated)

    async def delete(self, memory_id: str, namespace: str) -> None:
        await self.db.delete(_text(namespace, "namespace"), memory_id)

    async def history(self, memory_id: str, namespace: str) -> builtins.list[dict]:
        return await self.db.history(_text(namespace, "namespace"), memory_id)

    async def aclose(self) -> None:
        try:
            close = getattr(self.inference, "aclose", None)
            if close:
                await close()
        finally:
            try:
                await self.embeddings.aclose()
            finally:
                await self.db.aclose()

    async def __aenter__(self) -> "AgentMemory":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
