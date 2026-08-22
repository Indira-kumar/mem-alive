import uuid

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory
from .store_interface import Store
from dataclasses import replace
from datetime import datetime, timezone


class SemanticStore(Store):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        top_k: int = 100,
        recall_threshold: float = 0.8,
        contradiction_threshold: float = 0.8,
        contradiction_check_k: int = 5
    ):
        super().__init__(embedding_provider=embedding_provider, db=db, recall_threshold=recall_threshold)
        self._over_fetch_k = max(100, top_k)
        self._contradiction_check_k = contradiction_check_k
        self._top_k = top_k
        self._contradiction_threshold = contradiction_threshold

    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int|None = None
    ) -> list[Memory] | None:
        search_query_vector = (await self._embedding_provider.embed([search_query]))[0]
        search_results = await self._db.search(
            namespace=namespace,
            vector=search_query_vector,
            top_k=self._over_fetch_k,
            metadata=metadata,
        )
        search_results = [search_result for search_result in search_results 
                          if search_result.score > self._recall_threshold 
                          and search_result.memory.superseded is False
                          and search_result.memory.memory_type=='semantic']
        top_k = top_k or self._top_k
        return [r.memory for r in search_results][:top_k]

    async def remember(self, namespace: str, fact: str, metadata: dict):
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        await self._supersede_similar_memory(namespace=namespace, fact_vector=fact_vector, metadata=metadata)
        id = str(uuid.uuid4())
        memory = Memory(
            id=id,
            content=fact,
            vector=fact_vector,
            metadata=metadata,
            namespace=namespace,
            memory_type="semantic",
        )
        await self._db.upsert(memory=memory)

    async def _supersede_similar_memory(self, namespace:str, fact_vector:list[float], metadata:dict):
        search_results = await self._db.search(namespace=namespace, vector=fact_vector, top_k=self._contradiction_check_k,
                                         metadata=metadata)
        search_results = [r for r in search_results if r.memory.superseded is False
                          and r.memory.memory_type=='semantic']
        if not search_results:
            return
        best_match = max(search_results, key=lambda r: r.score)
        if best_match.score > self._contradiction_threshold:
            updated_memory = replace(best_match.memory, superseded=True, updated_at=datetime.now(timezone.utc))
            await self._db.upsert(updated_memory)