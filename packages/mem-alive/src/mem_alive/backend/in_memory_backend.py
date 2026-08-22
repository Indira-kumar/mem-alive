from collections import defaultdict

from ..backend.storage_backend_interface import StorageBackend
from schema.memory_schema import Memory

import numpy as np
import heapq


class InMemoryBackend(StorageBackend):
    def __init__(self):
        # id to Memory Mapping
        self._data: dict[str, Memory] = {}

        # namespace to id mapping
        self._by_namespace: dict[str, set[str]] = defaultdict(set)

    def search(self, namespace:str, vector:list[float], metadata:dict, top_k:int = 50) -> list[Memory]:
        id_in_namespace = list(self._by_namespace.get(namespace, set()))
        candidates = [self._data[i] for i in id_in_namespace]
        if metadata:
            candidates = [c for c in candidates if self._matches_metadata(c, metadata)]
        if not candidates:
            return []

        query = np.array(vector, dtype=np.float32)
        matrix = np.array([c.vector for c in candidates], dtype=np.float32)

        query_unit = query / np.linalg.norm(query)
        matrix_unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        # need keepdims to keep the original dimension
        # equivalent to matrix_unit = [unit / np.linalg.norm(unit) for unit in matrix]

        cosine_similarity_scores = matrix_unit @ query_unit

        top = heapq.nlargest(top_k, zip(cosine_similarity_scores, candidates), key= lambda pair: pair[0])
        return [memory for _, memory in top]


        
    def get_memory_by_id(self, namespace:str, id: str) -> Memory|None:
        found_ids = self._by_namespace.get(namespace, set())
        if id in found_ids:
            return self._data[id]

    def upsert(self, memory: Memory) -> None:
        self._by_namespace[memory.namespace].add(memory.id)
        self._data[memory.id] = memory

    def delete(self, namespace:str,  id: str):
        found_ids = self._by_namespace[namespace]
        if id in found_ids:
            self._by_namespace[namespace].remove(id)
            del self._data[id]

    def _matches_metadata(self, memory: Memory, metadata:dict):
        return all(memory.metadata.get(k) == v for k, v in metadata.items())