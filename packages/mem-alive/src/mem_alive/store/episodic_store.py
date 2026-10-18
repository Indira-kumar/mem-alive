from datetime import datetime, timezone
from uuid import uuid4

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory, SearchResult
from .store_interface import Store


class EpisodicStore(Store):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        over_fetch_k: int = 100,
        recall_threshold: float = 0.7,
        half_life_hours: float = 36,
    ):
        super().__init__(
            embedding_provider=embedding_provider, db=db, recall_threshold=recall_threshold
        )
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self._over_fetch_k = over_fetch_k
        self._half_life_hours = half_life_hours

    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int | None = None
    ) -> list[Memory]:
        return [
            result.memory
            for result in await self.recall_scored(namespace, search_query, metadata, top_k)
        ]

    async def recall_scored(
        self, namespace: str, search_query: str, metadata: dict, top_k: int | None = None
    ) -> list[SearchResult]:
        search_query_vector = (await self._embedding_provider.embed([search_query]))[0]
        search_results = await self._db.search(
            namespace=namespace,
            vector=search_query_vector,
            metadata=metadata,
            top_k=max(self._over_fetch_k, top_k or 0),
            memory_type="episodic",
        )

        # filtering unrelated memory before decay gets applied.
        search_results = [
            r
            for r in search_results
            if r.score > self._recall_threshold
        ]
        decayed_results = []
        for r in search_results:
            age_seconds = max(
                0.0, (datetime.now(timezone.utc) - r.memory.created_at).total_seconds()
            )
            decay = 0.5 ** (
                age_seconds / (self._half_life_hours * 3600)
            )
            blended_score = r.score * decay
            decayed_results.append(SearchResult(memory=r.memory, score=blended_score))
        decayed_results.sort(key=lambda r: r.score, reverse=True)
        return decayed_results[:top_k]

    async def remember(self, namespace: str, fact: str, metadata: dict) -> Memory:
        id = str(uuid4())
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        memory = Memory(
            id=id,
            namespace=namespace,
            content=fact,
            vector=fact_vector,
            metadata=dict(metadata),
            memory_type="episodic",
        )
        await self._db.upsert(memory=memory)
        return memory
