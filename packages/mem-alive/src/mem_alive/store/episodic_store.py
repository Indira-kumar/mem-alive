from datetime import datetime, timezone
from .store_interface import Store
from ..embedding.embedding_provider import EmbeddingProvider
from ..backend.storage_backend_interface import StorageBackend
from ..schema.memory_schema import Memory
from uuid import uuid4

class EpisodicStore(Store):
    def __init__(self, embedding_provider: EmbeddingProvider, db:StorageBackend, over_fetch_k: int = 100,
                 recall_threshold: float = 0.8, half_life_hours:float = 36):
        self._embedding_provider = embedding_provider
        self._db = db
        self._over_fetch_k = over_fetch_k
        self._recall_threshold = recall_threshold
        self._half_life_hours = half_life_hours

    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int|None = None
    ) -> list[Memory] | None:
        search_query_vector = (await self._embedding_provider.embed([search_query]))[0]
        search_results = await self._db.search(namespace=namespace, vector=search_query_vector, metadata=metadata, 
                                         top_k=self._over_fetch_k)

        # filtering unrelated memory before decay gets applied.
        search_results = [r for r in search_results if r.score > self._recall_threshold]
        decayed_results = []
        for r in search_results:
            decay = 0.5 ** ((datetime.now(timezone.utc) - r.memory.created_at).total_seconds()/ (self._half_life_hours * 3600))
            blended_score = r.score * decay
            decayed_results.append((blended_score, r.memory))
        decayed_results.sort(key= lambda r: r[0], reverse=True)
        return [r[1] for r in decayed_results[:top_k]]

    async def remember(self, namespace: str, fact: str, metadata: dict) -> None:
        id = str(uuid4())
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        memory = Memory(id=id, namespace=namespace, content=fact, vector=fact_vector, metadata=metadata, 
                        memory_type='episodic')
        await self._db.upsert(memory=memory)
