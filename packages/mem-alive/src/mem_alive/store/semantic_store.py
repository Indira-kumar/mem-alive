import uuid

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory
from .store_interface import Store


class SemanticStore(Store):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        top_k: int = 100,
        similarity_threshold: float = 0.8,
    ):
        self._embedding_provider = embedding_provider
        self._db = db
        self._over_fetch_k = max(100, top_k)
        self._top_k = top_k
        # TODO: to implement threshold by introducing similarity score as part of search results
        self._similarity_threshold = similarity_threshold

    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int|None = None
    ) -> list[Memory] | None:
        search_query_vector = (await self._embedding_provider.embed([search_query]))[0]
        memories = self._db.search(
            namespace=namespace,
            vector=search_query_vector,
            top_k=self._over_fetch_k,
            metadata=metadata,
        )
        top_k = top_k or self._top_k
        return [memory for memory in memories if not memory.superseded][:top_k]

    async def remember(self, namespace: str, fact: str, metadata: dict):
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        id = str(uuid.uuid4())
        memory = Memory(
            id=id,
            content=fact,
            vector=fact_vector,
            metadata=metadata,
            namespace=namespace,
            memory_type="semantic",
        )
        self._db.upsert(memory=memory)
