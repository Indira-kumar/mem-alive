from abc import ABC, abstractmethod

from ..backend.storage_backend_interface import StorageBackend
from ..embedding.embedding_provider import EmbeddingProvider
from ..schema.memory_schema import Memory


class Store(ABC):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db: StorageBackend,
        recall_threshold: float = 0.8,
    ):
        self._embedding_provider = embedding_provider
        self._db = db
        self._recall_threshold = recall_threshold

    @abstractmethod
    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int | None = None
    ) -> list[Memory]:
        pass

    @abstractmethod
    async def remember(self, namespace: str, fact: str, metadata: dict) -> None:
        pass
