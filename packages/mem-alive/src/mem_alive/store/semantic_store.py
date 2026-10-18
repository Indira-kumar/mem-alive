import uuid

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory, SearchResult
from .store_interface import Store


class SemanticStore(Store):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        top_k: int = 100,
        recall_threshold: float = 0.7,
    ):
        super().__init__(
            embedding_provider=embedding_provider, db=db, recall_threshold=recall_threshold
        )
        self._over_fetch_k = max(100, top_k)
        self._top_k = top_k

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
            top_k=max(self._over_fetch_k, top_k or 0),
            metadata=metadata,
            memory_type="semantic",
        )
        search_results = [
            search_result
            for search_result in search_results
            if search_result.score > self._recall_threshold
            and search_result.memory.superseded is False
        ]
        top_k = self._top_k if top_k is None else top_k
        return search_results[:top_k]

    async def remember(self, namespace: str, fact: str, metadata: dict) -> Memory:
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        memory = Memory(
            id=str(uuid.uuid4()),
            content=fact,
            vector=fact_vector,
            metadata=dict(metadata),
            namespace=namespace,
            memory_type="semantic",
        )
        await self._db.upsert(memory=memory)
        return memory
