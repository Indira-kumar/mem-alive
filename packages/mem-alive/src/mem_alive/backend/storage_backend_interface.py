from abc import ABC, abstractmethod

from ..schema.memory_schema import Memory, MemoryEvent, SearchResult


class StorageBackend(ABC):
    @abstractmethod
    async def search(
        self,
        namespace: str,
        vector: list[float],
        metadata: dict,
        top_k: int = 50,
        memory_type: str | None = None,
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

    @abstractmethod
    async def purge(self, namespace: str, id: str) -> None:
        pass

    @abstractmethod
    async def history(self, namespace: str, id: str) -> list[MemoryEvent]:
        pass

    @abstractmethod
    async def list_memories(
        self,
        namespace: str,
        metadata: dict,
        memory_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        pass
