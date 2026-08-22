from uuid import uuid4

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory
from .store_interface import Store


class ProceduralStore(Store):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        over_fetch_k: int = 100,
        recall_threshold: float = 0.8,
        keyword_weight: float = 0.5,
    ):
        super().__init__(
            embedding_provider=embedding_provider, db=db, recall_threshold=recall_threshold
        )
        self._over_fetch_k = over_fetch_k
        self._keyword_weight = keyword_weight

    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int | None = None
    ):
        search_query_vector = (await self._embedding_provider.embed([search_query]))[0]
        candidates = await self._db.search(
            namespace=namespace,
            vector=search_query_vector,
            metadata=metadata,
            top_k=self._over_fetch_k,
        )
        candidates = [
            r
            for r in candidates
            if r.score > self._recall_threshold and r.memory.memory_type == "procedural"
        ]
        results = []
        for r in candidates:
            keyword_score = self._keyword_score(search_query=search_query, content=r.memory.content)
            blended_score = (
                self._keyword_weight * keyword_score + (1 - self._keyword_weight) * r.score
            )
            results.append((blended_score, r.memory))
        results.sort(key=lambda r: r[0], reverse=True)
        return [r[1] for r in results[:top_k]]

    async def remember(self, namespace: str, fact: str, metadata: dict):
        fact_vector = (await self._embedding_provider.embed([fact]))[0]
        memory = Memory(
            id=str(uuid4()),
            content=fact,
            vector=fact_vector,
            namespace=namespace,
            metadata=metadata,
            memory_type="procedural",
        )
        await self._db.upsert(memory=memory)

    def _keyword_score(self, search_query: str, content: str):
        # TODO: Improve to a better algo like BM25
        query_tokens = set(search_query.lower().split())
        content_tokens = set(content.lower().split())

        if not query_tokens:
            return 0.0
        overlap = query_tokens & content_tokens
        return len(overlap) / len(query_tokens)
