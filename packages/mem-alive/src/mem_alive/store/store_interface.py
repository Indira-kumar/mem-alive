from abc import ABC, abstractmethod

from ..schema.memory_schema import Memory


class Store(ABC):
    @abstractmethod
    async def recall(
        self, namespace: str, search_query: str, metadata: dict, top_k: int|None = None
    ) -> list[Memory] | None:
        pass

    @abstractmethod
    async def remember(self, namespace: str, fact: str, metadata: dict) -> None:
        pass
