import heapq
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from ..backend.storage_backend_interface import StorageBackend
from ..schema.memory_schema import Memory, MemoryEvent, SearchResult


class InMemoryBackend(StorageBackend):
    def __init__(self):
        self._data: dict[tuple[str, str], Memory] = {}

        # namespace to id mapping
        self._by_namespace: dict[str, set[str]] = defaultdict(set)
        self._history: dict[tuple[str, str], list[MemoryEvent]] = defaultdict(list)

    async def search(
        self,
        namespace: str,
        vector: list[float],
        metadata: dict,
        top_k: int = 50,
        memory_type: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("query vector must be nonempty and finite")
        id_in_namespace = list(self._by_namespace.get(namespace, set()))
        candidates = [self._data[namespace, i] for i in id_in_namespace]
        if memory_type is not None:
            candidates = [
                candidate for candidate in candidates if candidate.memory_type == memory_type
            ]
        if metadata:
            candidates = [c for c in candidates if self._matches_metadata(c, metadata)]
        if not candidates:
            return []

        query = np.array(vector, dtype=np.float32)
        matrix = np.array([c.vector for c in candidates], dtype=np.float32)
        if query.ndim != 1 or query.size != matrix.shape[1]:
            raise ValueError("query vector dimension does not match stored vectors")

        query_norm = np.linalg.norm(query)
        row_norms = np.linalg.norm(matrix, axis=1)
        query_unit = query / query_norm if query_norm else np.zeros_like(query)
        matrix_unit = np.divide(
            matrix,
            row_norms[:, None],
            out=np.zeros_like(matrix),
            where=row_norms[:, None] != 0,
        )

        cosine_similarity_scores = matrix_unit @ query_unit

        top = heapq.nlargest(
            top_k, zip(cosine_similarity_scores, candidates), key=lambda pair: pair[0]
        )
        return [SearchResult(memory=deepcopy(memory), score=float(score)) for score, memory in top]

    async def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        found_ids = self._by_namespace.get(namespace, set())
        if id in found_ids:
            return deepcopy(self._data[namespace, id])
        return None

    async def upsert(self, memory: Memory) -> None:
        if not memory.vector or not all(math.isfinite(value) for value in memory.vector):
            raise ValueError("memory vector must be nonempty and finite")
        existing_ids = self._by_namespace[memory.namespace]
        other_id = next((id for id in existing_ids if id != memory.id), None)
        if other_id is not None and len(self._data[memory.namespace, other_id].vector) != len(
            memory.vector
        ):
            raise ValueError("memory vector dimension does not match namespace")
        key = (memory.namespace, memory.id)
        action = "update" if key in self._data else "add"
        self._by_namespace[memory.namespace].add(memory.id)
        self._data[key] = deepcopy(memory)
        self._history[key].append(MemoryEvent(action, deepcopy(memory), datetime.now(timezone.utc)))

    async def delete(self, namespace: str, id: str):
        found_ids = self._by_namespace.get(namespace, set())
        if id in found_ids:
            memory = self._data[namespace, id]
            self._history[namespace, id].append(
                MemoryEvent("delete", deepcopy(memory), datetime.now(timezone.utc))
            )
            self._by_namespace[namespace].remove(id)
            del self._data[namespace, id]

    async def purge(self, namespace: str, id: str) -> None:
        self._by_namespace.get(namespace, set()).discard(id)
        self._data.pop((namespace, id), None)
        self._history.pop((namespace, id), None)

    async def history(self, namespace: str, id: str) -> list[MemoryEvent]:
        return deepcopy(self._history.get((namespace, id), []))

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
        records = (
            self._data[namespace, id] for id in self._by_namespace.get(namespace, set())
        )
        filtered = [
            record
            for record in records
            if not record.superseded
            and (memory_type is None or record.memory_type == memory_type)
            and self._matches_metadata(record, metadata)
        ]
        filtered.sort(key=lambda record: (record.created_at, record.id), reverse=True)
        return deepcopy(filtered[offset : offset + limit])

    def _matches_metadata(self, memory: Memory, metadata: dict):
        return all(memory.metadata.get(k) == v for k, v in metadata.items())
