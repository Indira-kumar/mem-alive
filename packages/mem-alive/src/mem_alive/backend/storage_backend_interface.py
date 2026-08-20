from abc import ABC, abstractmethod
from ..schema.memory_schema import Memory

class StorageBackend(ABC):

    @abstractmethod
    def search(self, vector:list[float], metadata:dict, namespace: str, top_k:int = 50) -> list[Memory]:
        pass

    @abstractmethod
    def get_memory_by_id(self, id: str) -> Memory:
        pass

    @abstractmethod
    def upsert(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def delete(self, id: str, namespace:str):
        pass