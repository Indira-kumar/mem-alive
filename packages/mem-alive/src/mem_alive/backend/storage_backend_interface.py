from abc import ABC, abstractmethod

from ..schema.memory_schema import Memory, SearchResult


class StorageBackend(ABC):
    @abstractmethod
    async def search(
        self, namespace: str, vector: list[float], metadata: dict, top_k: int = 50
    ) -> list[SearchResult]:
        pass

    @abstractmethod
    async def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        pass

    @abstractmethod
    async def upsert(self, memory: Memory) -> None:
        pass

    @abstractmethod
    async def delete(self, namespace: str, id: str):
        pass
