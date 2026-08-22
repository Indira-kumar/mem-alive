import asyncio

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory as MemoryRecord
from ..store.episodic_store import EpisodicStore
from ..store.procedural_store import ProceduralStore
from ..store.semantic_store import SemanticStore


class Memory:
    def __init__(self, embedding_provider: EmbeddingProvider, db: StorageBackend):
        self._stores = {
            "procedural": ProceduralStore(db=db, embedding_provider=embedding_provider),
            "semantic": SemanticStore(db=db, embedding_provider=embedding_provider),
            "episodic": EpisodicStore(db=db, embedding_provider=embedding_provider),
        }

    async def recall(
        self,
        namespace: str,
        search_query: str,
        metadata: dict,
        top_k: int | None = None,
        memory_type: str | None = None,
    ) -> list[MemoryRecord] | None:
        metadata = metadata or {}
        if memory_type:
            return await self._stores[memory_type].recall(
                namespace=namespace, search_query=search_query, metadata=metadata, top_k=top_k
            )
        results = await asyncio.gather(
            *(
                store.recall(namespace, search_query, metadata, top_k)
                for store in self._stores.values()
            )
        )
        return [m for group in results for m in group]

    async def remember(self, memory_type: str, namespace: str, fact: str, metadata: dict) -> None:
        await self._stores[memory_type].remember(namespace=namespace, fact=fact, metadata=metadata)
