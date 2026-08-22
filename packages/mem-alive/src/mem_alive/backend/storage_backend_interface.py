from abc import ABC, abstractmethod

from ..schema.memory_schema import Memory


class StorageBackend(ABC):
    @abstractmethod
    def search(
        self, namespace: str, vector: list[float], metadata: dict, top_k: int = 50
    ) -> list[Memory]:
        pass

    @abstractmethod
    def get_memory_by_id(self, namespace: str, id: str) -> Memory | None:
        pass

    @abstractmethod
    def upsert(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def delete(self, namespace: str, id: str):
        pass
