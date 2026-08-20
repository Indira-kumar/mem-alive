from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):

    @abstractmethod
    def embed(text: list[str]) -> list[list[float]]:
        pass